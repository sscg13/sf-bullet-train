use bullet::{
    game::{
        formats::sfbinpack::{
            chess::{piecetype::PieceType, r#move::MoveType},
            TrainingDataEntry,
        },
        inputs::SparseInputType,
        outputs::{self, OutputBuckets},
    },
    nn::{
        optimiser::{Ranger, RangerParams},
        Affine, InitSettings, ModelNode, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader, ValueTrainerBuilder},
};
use bulletformat::ChessBoard;

use crate::full_threats::ThreatInputsBucketsMirrored;
mod full_threats;

#[derive(Clone, Copy, Default)]
pub struct SfMaterialCount;
impl OutputBuckets<ChessBoard> for SfMaterialCount {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        let piece_count = pos.occ().count_ones() as u8 - 1;
        (piece_count / 4) as u8
    }
}

const L1: usize = 256;
const L2: usize = 31;
const L3: usize = 32;

const FT_QUANTIZED_ONE: i16 = 256;
const FT_QUANTIZED_MAX: i16 = 255;
const HIDDEN_QUANTIZED_ONE: i16 = 128;
const HIDDEN_QUANTIZED_MAX: i16 = 127;
const WEIGHT_SCALE_L1: i16 = 128;
const WEIGHT_SCALE_L2: i16 = 64;
const WEIGHT_SCALE_OUT: i16 = 128;
const PSQT_SCALE: i32 = 600 * 16;
const OUTPUT_DENOMINATOR: i32 = HIDDEN_QUANTIZED_ONE as i32 * WEIGHT_SCALE_OUT as i32 * 2;
const FAKE_QUANTIZE_EPS: f32 = 1e-5;

const FT_ACT_MAX: f32 = FT_QUANTIZED_MAX as f32 / FT_QUANTIZED_ONE as f32;
const HIDDEN_ACT_MAX: f32 = HIDDEN_QUANTIZED_MAX as f32 / HIDDEN_QUANTIZED_ONE as f32;

/// assumes the ThreatInputsBucketsMirrored input type, and that `weights` contains first the factoriser weights, and then the rest
fn merge_factoriser(weights: &[f32], output_size: usize) -> Vec<f32> {
    let factorised_end = output_size * ThreatInputsBucketsMirrored::FACTORISER_SIZE;
    let factorised_weights = &weights[0..factorised_end];

    let feature_end = factorised_end
        + output_size
            * (ThreatInputsBucketsMirrored::HALFKA_V2_SIZE
                + ThreatInputsBucketsMirrored::THREATS_SIZE);
    let feature_weights = &weights[factorised_end..feature_end];

    (0..output_size
        * (ThreatInputsBucketsMirrored::HALFKA_V2_SIZE + ThreatInputsBucketsMirrored::THREATS_SIZE))
        .map(|idx| {
            let feature = idx / output_size;
            if feature >= ThreatInputsBucketsMirrored::HALFKA_V2_SIZE {
                feature_weights[idx]
            } else {
                let l1 = idx % output_size;
                let factorised_feature =
                    ThreatInputsBucketsMirrored::derive_factorised_feature(feature);
                feature_weights[idx] + factorised_weights[factorised_feature * output_size + l1]
            }
        })
        .collect::<Vec<f32>>()
}

fn quantised_affine_forward<'a>(
    layer: Affine<'a>,
    input: ModelNode<'a>,
    weight_scale: i16,
    bias_scale: i32,
) -> ModelNode<'a> {
    layer
        .weights
        .faux_quantise(weight_scale as f32, true)
        .matmul(input)
        + layer.bias.faux_quantise(bias_scale as f32, true)
}

fn main() {
    let inputs = ThreatInputsBucketsMirrored::default();

    let output_buckets = SfMaterialCount::default();
    const NUM_OUTPUT_BUCKETS: usize = <SfMaterialCount as outputs::OutputBuckets<_>>::BUCKETS;

    let saved_format = vec![
        SavedFormat::id("l0b")
            .round()
            .quantise::<i16>(FT_QUANTIZED_ONE),
        // weights
        SavedFormat::id("l0w")
            .transform(move |_, weights| merge_factoriser(&weights, L1))
            .round()
            .quantise::<i16>(FT_QUANTIZED_ONE),
        SavedFormat::id("pst")
            .transform(move |_, weights| merge_factoriser(&weights, NUM_OUTPUT_BUCKETS))
            .round()
            .quantise::<i32>(PSQT_SCALE),
        SavedFormat::id("l1b")
            .round()
            .quantise::<i32>((WEIGHT_SCALE_L1 as i32) * (HIDDEN_QUANTIZED_ONE as i32)),
        SavedFormat::id("l1w")
            .round()
            .quantise::<i8>(WEIGHT_SCALE_L1)
            .transpose(),
        SavedFormat::id("l2b")
            .round()
            .quantise::<i32>((WEIGHT_SCALE_L2 as i32) * (HIDDEN_QUANTIZED_ONE as i32)),
        SavedFormat::id("l2w")
            .round()
            .quantise::<i8>(WEIGHT_SCALE_L2)
            .transpose(),
        SavedFormat::id("l3b")
            .round()
            .quantise::<i32>((WEIGHT_SCALE_OUT as i32) * (HIDDEN_QUANTIZED_ONE as i32)),
        SavedFormat::id("l3w")
            .round()
            .quantise::<i8>(WEIGHT_SCALE_OUT)
            .transpose(),
    ];

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(Ranger)
        .loss_fn(|output, targets| output.sigmoid().power_error(targets, 2.6))
        .inputs(inputs)
        .output_buckets(output_buckets)
        .save_format(saved_format.as_slice())
        .build(|builder, stm, ntm, buckets| {
            // trainable weights
            let l0 = builder.new_affine("l0", inputs.num_inputs(), L1);
            let l1 = builder.new_affine("l1", L1, NUM_OUTPUT_BUCKETS * (L2 + 1));
            // let l1_fact = builder.new_affine("l1_fact", L1, L2 + 1);
            let l2 = builder.new_affine("l2", L2 * 2, NUM_OUTPUT_BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, NUM_OUTPUT_BUCKETS);
            let pst = builder.new_weights(
                "pst",
                Shape::new(
                    NUM_OUTPUT_BUCKETS,
                    ThreatInputsBucketsMirrored::FACTORISER_SIZE
                        + ThreatInputsBucketsMirrored::HALFKA_V2_SIZE
                        + ThreatInputsBucketsMirrored::THREATS_SIZE,
                ),
                InitSettings::Zeroed,
            );

            // inference
            let stm_subnet =
                (quantised_affine_forward(l0, stm, FT_QUANTIZED_ONE, FT_QUANTIZED_ONE as i32)
                    .clip_pass_through_grad(0.0, FT_ACT_MAX))
                .pairwise_mul()
                .faux_quantise(HIDDEN_QUANTIZED_ONE as f32, false);
            let ntm_subnet =
                (quantised_affine_forward(l0, ntm, FT_QUANTIZED_ONE, FT_QUANTIZED_ONE as i32)
                    .clip_pass_through_grad(0.0, FT_ACT_MAX))
                .pairwise_mul()
                .faux_quantise(HIDDEN_QUANTIZED_ONE as f32, false);
            let mut out = stm_subnet.concat(ntm_subnet);

            out = quantised_affine_forward(
                l1,
                out,
                WEIGHT_SCALE_L1,
                (WEIGHT_SCALE_L1 as i32) * (HIDDEN_QUANTIZED_ONE as i32),
            )
            .select(buckets); // + l1_fact.forward(out);

            let skip_neuron = out.slice_rows(15, 16);
            out = out.slice_rows(0, 15);

            let squared = (out.abs_pow(2.0) + FAKE_QUANTIZE_EPS)
                .faux_quantise(HIDDEN_QUANTIZED_ONE as f32, false);
            let linear =
                (out + FAKE_QUANTIZE_EPS).faux_quantise(HIDDEN_QUANTIZED_ONE as f32, false);
            out = squared
                .concat(linear)
                .clip_pass_through_grad(0.0, HIDDEN_ACT_MAX);

            out = (quantised_affine_forward(
                l2,
                out,
                WEIGHT_SCALE_L2,
                (WEIGHT_SCALE_L2 as i32) * (HIDDEN_QUANTIZED_ONE as i32),
            )
            .select(buckets)
                + FAKE_QUANTIZE_EPS)
                .faux_quantise(HIDDEN_QUANTIZED_ONE as f32, false)
                .clip_pass_through_grad(0.0, HIDDEN_ACT_MAX);
            out = quantised_affine_forward(
                l3,
                out,
                WEIGHT_SCALE_OUT,
                (WEIGHT_SCALE_OUT as i32) * (HIDDEN_QUANTIZED_ONE as i32),
            )
            .select(buckets);

            let stm_pst = pst.matmul(stm).select(buckets);
            let ntm_pst = pst.matmul(ntm).select(buckets);
            let pst_out = 0.5 * stm_pst - 0.5 * ntm_pst;
            out = (out + skip_neuron)
                .faux_quantise(OUTPUT_DENOMINATOR as f32, true)
                .faux_quantise(PSQT_SCALE as f32, false)
                + pst_out;

            out
        });

    trainer.optimiser.set_params_for_weight(
        "l3w",
        RangerParams {
            min_weight: -(HIDDEN_QUANTIZED_MAX as f32 / WEIGHT_SCALE_OUT as f32),
            max_weight: HIDDEN_QUANTIZED_MAX as f32 / WEIGHT_SCALE_OUT as f32,
            ..Default::default()
        },
    );

    let num_params: usize = trainer
        .optimiser
        .cpu_weights()
        .unwrap()
        .iter()
        .map(|(_, weights)| weights.shape.size())
        .sum();
    println!("Params: {num_params}");

    let schedule = TrainingSchedule {
        net_id: "test".to_string(),
        eval_scale: 600.0,
        steps: TrainingSteps {
            batch_size: 16_384,
            batches_per_superbatch: 1024,
            start_superbatch: 1,
            end_superbatch: 150,
        },
        wdl_scheduler: wdl::ConstantWDL { value: 0.0 },
        lr_scheduler: lr::StepLR {
            start: 0.001,
            gamma: 0.3,
            step: 60,
        },
        save_rate: 150,
    };

    let settings = LocalSettings {
        threads: 4,
        test_set: None,
        output_directory: "checkpoints",
        batch_queue_size: 512,
    };

    let data_loader = {
        let file_path = "small.binpack";
        let buffer_size_mb = 1024;
        let threads = 8;
        fn filter(entry: &TrainingDataEntry) -> bool {
            entry.ply >= 16
                && !entry.pos.is_checked(entry.pos.side_to_move())
                && entry.score.unsigned_abs() <= 10000
                && entry.mv.mtype() == MoveType::Normal
                && entry.pos.piece_at(entry.mv.to()).piece_type() == PieceType::None
        }

        loader::SfBinpackLoader::new(file_path, buffer_size_mb, threads, filter)
    };
    //trainer.profile_all_nodes();
    trainer.run(&schedule, &settings, &data_loader);
    // trainer.load_from_checkpoint("checkpoints/test-1");
    //trainer.report_profiles();
    let eval =
        600.0 * trainer.eval("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
    let eval =
        600.0 * trainer.eval("r1bq1rk1/ppppbppp/3n4/4R3/8/8/PPPP1PPP/RNBQ1BK1 w - - 1 9 | 0 | 0.0");
    println!("Eval: {eval:.3}cp");
}
