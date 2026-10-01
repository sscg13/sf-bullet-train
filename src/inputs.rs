use bullet::game::{
    inputs::{Chess768, Factorised, Factorises, SparseInputType},
    outputs::OutputBuckets,
};
use bulletformat::ChessBoard;

pub const PLANES: usize = 11 * 64;
pub const NUM_INPUTS: usize = 64 * PLANES;
pub type InputFeatures = Factorised<ShatranjInputs, Chess768>;

// Bullet stores P,N,B,R,Q,K. Fairy-Stockfish enumerates the variant's
// non-king types by PieceType: P,N,R,FERS,ALFIL, then the shared king plane.
const PIECE_ORDER: [usize; 6] = [0, 1, 4, 2, 3, 5];
const BULLET_ORDER: [usize; 6] = [0, 1, 3, 4, 2, 5];

#[derive(Clone, Copy, Default)]
pub struct ShatranjInputs;

impl ShatranjInputs {
    pub fn index(perspective: usize, square: usize, piece: u8, king: usize) -> usize {
        let colour = usize::from(piece & 8 != 0) ^ perspective;
        let kind = PIECE_ORDER[usize::from(piece & 7)];
        let plane = (2 * kind + colour).min(10);
        (square ^ (56 * perspective)) + 64 * plane + PLANES * king
    }
}

impl SparseInputType for ShatranjInputs {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        NUM_INPUTS
    }
    fn max_active(&self) -> usize {
        32
    }
    fn shorthand(&self) -> String {
        "shatranj-HalfKAv2".into()
    }
    fn description(&self) -> String {
        "Fairy-Stockfish Shatranj HalfKAv2, 64 king squares, no mirroring".into()
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &ChessBoard, mut f: F) {
        // ChessBoard already orients squares and colours to the side to move.
        // opp_ksq is already rank-flipped to the opponent's perspective.
        for (piece, square) in *pos {
            f(
                Self::index(0, square.into(), piece, pos.our_ksq().into()),
                Self::index(1, square.into(), piece, pos.opp_ksq().into()),
            );
        }
    }
}

impl Factorises<ShatranjInputs> for Chess768 {
    fn derive_feature(&self, _: &ShatranjInputs, feature: usize) -> Option<usize> {
        let square = feature % 64;
        let plane = feature % PLANES / 64;
        let king = feature / PLANES;
        let kind = BULLET_ORDER[plane / 2];
        let colour = if plane == 10 {
            usize::from(square != king)
        } else {
            plane % 2
        };
        Some(384 * colour + 64 * kind + square)
    }
}

#[derive(Clone, Copy, Default)]
pub struct MaterialCount;

impl OutputBuckets<ChessBoard> for MaterialCount {
    const BUCKETS: usize = 8;

    fn bucket(&self, pos: &ChessBoard) -> u8 {
        // Shatranj's initial position has 32 pieces; matches nnueMaxPieces.
        ((pos.occ().count_ones().saturating_sub(1) * 8 / 32).min(7)) as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piece_planes_and_black_orientation() {
        for (kind, order) in PIECE_ORDER.into_iter().enumerate() {
            for colour in 0..2 {
                for perspective in 0..2 {
                    for king in 0..64 {
                        for square in 0..64 {
                            let pc = kind as u8 | ((colour as u8) << 3);
                            let idx = ShatranjInputs::index(perspective, square, pc, king);
                            let plane = (2 * order + (colour ^ perspective)).min(10);
                            assert_eq!(
                                idx,
                                (square ^ (56 * perspective)) + 64 * plane + 704 * king
                            );
                            assert!(idx < NUM_INPUTS);
                        }
                    }
                }
            }
        }
        assert_eq!(ShatranjInputs::index(0, 2, 2, 3), 3 * 704 + 8 * 64 + 2); // alfil
        assert_eq!(ShatranjInputs::index(0, 4, 4, 3), 3 * 704 + 6 * 64 + 4); // fers
    }

    #[test]
    fn factoriser_matches_chess768_in_both_perspectives() {
        for fen in [
            "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR w - - 0 1",
            "rnbkqbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBKQBNR b - - 0 1",
            "7k/8/3q4/5b2/2B5/4Q3/8/K7 b - - 0 1",
        ] {
            let board: ChessBoard = format!("{fen} | 0 | 0.5").parse().unwrap();
            let mut real = Vec::new();
            let mut virtuals = Vec::new();
            ShatranjInputs.map_features(&board, |a, b| real.push((a, b)));
            Chess768.map_features(&board, |a, b| virtuals.push((a, b)));
            for ((a, b), expected) in real.into_iter().zip(virtuals) {
                assert_eq!(
                    (
                        Chess768.derive_feature(&ShatranjInputs, a).unwrap(),
                        Chess768.derive_feature(&ShatranjInputs, b).unwrap()
                    ),
                    expected
                );
            }
        }
    }

    #[test]
    fn every_material_bucket() {
        for pieces in 2..=32 {
            let board = ChessBoard {
                occ: (1_u64 << pieces) - 1,
                ..Default::default()
            };
            assert_eq!(MaterialCount.bucket(&board), ((pieces - 1) / 4) as u8);
        }
    }

    #[test]
    #[ignore = "requires FSF_PROBE and FSF_NETWORK; see README.md"]
    fn feature_indices_match_fairy_stockfish() {
        use std::{
            io::Write,
            process::{Command, Stdio},
        };
        let records: Vec<_> = include_str!("../tests/fixtures/shatranj.txt")
            .lines()
            .collect();
        let fens: Vec<_> = records
            .iter()
            .map(|line| line.split('|').next().unwrap().trim())
            .collect();
        let mut probe = Command::new(std::env::var("FSF_PROBE").expect("set FSF_PROBE"))
            .arg(std::env::var("FSF_NETWORK").expect("set FSF_NETWORK"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        writeln!(probe.stdin.take().unwrap(), "{}", fens.join("\n")).unwrap();
        let output = probe.wait_with_output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert_eq!(stdout.lines().count(), records.len());
        for ((record, fen), json) in records.iter().zip(fens).zip(stdout.lines()) {
            let board = crate::data::parse_record(record).unwrap();
            let mut ours = Vec::new();
            let mut theirs = Vec::new();
            ShatranjInputs.map_features(&board, |a, b| {
                ours.push(a);
                theirs.push(b);
            });
            let indices = |name| -> Vec<usize> {
                json.split(&format!("\"{name}\":["))
                    .nth(1)
                    .unwrap()
                    .split(']')
                    .next()
                    .unwrap()
                    .split(',')
                    .map(|v| v.parse().unwrap())
                    .collect()
            };
            let mut white = indices("white");
            let mut black = indices("black");
            ours.sort_unstable();
            theirs.sort_unstable();
            white.sort_unstable();
            black.sort_unstable();
            let (expected_ours, expected_theirs) = if fen.split_whitespace().nth(1) == Some("w") {
                (white, black)
            } else {
                (black, white)
            };
            assert_eq!(ours, expected_ours, "{fen}");
            assert_eq!(theirs, expected_theirs, "{fen}");
        }
    }
}
