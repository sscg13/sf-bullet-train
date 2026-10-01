mod cli;
mod data;
mod inputs;

#[cfg(all(feature = "cpu", feature = "cuda"))]
compile_error!("choose one backend: use --no-default-features --features cpu for CPU");
#[cfg(not(any(feature = "cpu", feature = "cuda")))]
compile_error!("enable either the cuda or cpu feature");

use bullet::{
    game::inputs::{Chess768, SparseInputType},
    nn::{
        optimiser::{Ranger, RangerParams},
        InitSettings, Shape,
    },
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};
use inputs::{InputFeatures, MaterialCount, ShatranjInputs};

const L1: usize = 512;
const L2: usize = 16;
const L3: usize = 32;
const BUCKETS: usize = 8;
const OUTPUT_SCALE: f32 = 9600.0 / 127.0;

fn main() {
    if let Err(error) = run() {
        eprintln!("Error: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", cli::HELP);
        return Ok(());
    }
    if args[0] == "prepare" {
        if args.len() != 3 {
            return Err("usage: sf-bullet-train prepare INPUT.txt OUTPUT.bf".into());
        }
        let count = data::prepare(args[1].as_ref(), args[2].as_ref())?;
        println!("Wrote {count} Shatranj records to {}", args[2]);
        return Ok(());
    }
    let config = cli::Config::parse(&args)?;
    let metadata = std::fs::metadata(&config.data).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() % 32 != 0 {
        return Err(
            "training data must be nonempty 32-byte Bullet records; use prepare for FEN text"
                .into(),
        );
    }
    if config.data.ends_with(".binpack") {
        return Err("orthodox Stockfish binpack is not a Shatranj data format; use prepare".into());
    }
    std::fs::create_dir_all(&config.output).map_err(|e| e.to_string())?;

    let inputs = InputFeatures::from_parts(ShatranjInputs, Chess768);
    let num_inputs = inputs.num_inputs();
    // This intermediate file has tensors in Bullet order. The Python converter
    // writes the engine header, interleaves stacks, and pads dense-layer inputs.
    let saved_format = vec![
        SavedFormat::id("l0b").round().quantise::<i16>(127),
        SavedFormat::id("l0w")
            .transform(move |_, w| inputs.merge_factoriser(w))
            .round()
            .quantise::<i16>(127),
        SavedFormat::id("pst")
            .transform(move |_, w| inputs.merge_factoriser(w))
            .round()
            .quantise::<i32>(9600),
        SavedFormat::id("l1b").round().quantise::<i32>(8128),
        SavedFormat::id("l1w")
            .transpose()
            .round()
            .quantise::<i8>(64),
        SavedFormat::id("l2b").round().quantise::<i32>(8128),
        SavedFormat::id("l2w")
            .transpose()
            .round()
            .quantise::<i8>(64),
        SavedFormat::id("l3b").round().quantise::<i32>(9600),
        // 9600/127 is fractional. Apply the ratio before integer quantisation.
        SavedFormat::id("l3w")
            .transpose()
            .transform(|_, w| w.into_iter().map(|v| v * OUTPUT_SCALE).collect())
            .round()
            .quantise::<i8>(1),
    ];

    let trainer_builder = ValueTrainerBuilder::default();
    #[cfg(feature = "cpu")]
    let trainer_builder = trainer_builder.use_devices(vec![()]);
    let mut trainer = trainer_builder
        .dual_perspective()
        .optimiser(Ranger)
        .loss_fn(|out, targets| {
            out.linear_comb(600.0 / 361.0, out, 0.0)
                .sigmoid()
                .power_error(targets, 2.0)
        })
        .inputs(inputs)
        .output_buckets(MaterialCount)
        .save_format(&saved_format)
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", num_inputs, L1);
            let l1 = builder.new_affine("l1", 2 * L1, BUCKETS * L2);
            let l2 = builder.new_affine("l2", L2, BUCKETS * L3);
            let l3 = builder.new_affine("l3", L3, BUCKETS);
            let pst =
                builder.new_weights("pst", Shape::new(BUCKETS, num_inputs), InitSettings::Zeroed);
            let out = l0.forward(stm).crelu().concat(l0.forward(ntm).crelu());
            let out = l1.forward(out).select(buckets).crelu();
            let out = l2.forward(out).select(buckets).crelu();
            let out = l3.forward(out).select(buckets);
            let psqt = pst.matmul(stm).select(buckets).linear_comb(
                0.5,
                pst.matmul(ntm).select(buckets),
                -0.5,
            );
            out + psqt
        });

    for (name, limit) in [
        ("l1w", 127.0 / 64.0),
        ("l2w", 127.0 / 64.0),
        ("l3w", 127.0 / OUTPUT_SCALE),
    ] {
        trainer.optimiser.set_params_for_weight(
            name,
            RangerParams {
                min_weight: -limit,
                max_weight: limit,
                ..Default::default()
            },
        );
    }
    if let Some(checkpoint) = &config.resume {
        trainer.load_from_checkpoint(checkpoint);
    }
    let schedule = TrainingSchedule {
        net_id: config.net_id.clone(),
        // variant-nnue-pytorch's target probability is sigmoid(score/410).
        eval_scale: 410.0,
        steps: TrainingSteps {
            batch_size: config.batch_size,
            batches_per_superbatch: config.batches,
            start_superbatch: config.start,
            end_superbatch: config.start + (config.superbatches - 1),
        },
        wdl_scheduler: wdl::ConstantWDL { value: config.wdl },
        lr_scheduler: lr::StepLR {
            start: config.learning_rate,
            gamma: 0.987,
            step: 1,
        },
        save_rate: config.save_rate,
    };
    let settings = LocalSettings {
        threads: config.threads,
        test_set: None,
        output_directory: &config.output,
        batch_queue_size: 4,
    };
    println!(
        "Shatranj HalfKAv2: {} inputs, (512x2)->16->32->1, 8 stacks",
        inputs::NUM_INPUTS
    );
    let loader = DirectSequentialDataLoader::new(&[&config.data]);
    trainer.run(&schedule, &settings, &loader);
    let output = format!("{}/{}-final.bin", config.output, config.net_id);
    trainer.save_quantised(&output).map_err(|e| e.to_string())?;
    println!("Export with: python convert_quantised_to_pytorch.py \"{output}\" shatranj.nnue");
    Ok(())
}
