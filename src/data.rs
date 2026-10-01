use bulletformat::ChessBoard;
use std::{
    fs::File,
    io::{BufRead, BufReader, BufWriter, Write},
    path::Path,
};

// Input scores and results are White-relative, as expected by ChessBoard.
// FEN B means ALFIL; Q means FERS. No chess move generation is used here.
pub fn parse_record(line: &str) -> Result<ChessBoard, String> {
    let fields: Vec<_> = line.split('|').map(str::trim).collect();
    if fields.len() != 3 {
        return Err("expected FEN | White score | White result".into());
    }
    let fen: Vec<_> = fields[0].split_whitespace().collect();
    if fen.len() != 6 || !["w", "b"].contains(&fen[1]) || fen[2] != "-" || fen[3] != "-" {
        return Err("expected six-field Shatranj FEN with no castling or en passant".into());
    }
    if fen[4].parse::<u32>().is_err() || !matches!(fen[5].parse::<u32>(), Ok(1..)) {
        return Err("invalid FEN move counters".into());
    }
    let ranks: Vec<_> = fen[0].split('/').collect();
    if ranks.len() != 8 {
        return Err("FEN must have eight ranks".into());
    }
    let mut kings = [0; 2];
    let mut pieces = 0;
    for rank in ranks {
        let mut files = 0;
        for c in rank.chars() {
            if ('1'..='8').contains(&c) {
                files += c.to_digit(10).unwrap();
            } else if "PNBRQKpnbrqk".contains(c) {
                files += 1;
                pieces += 1;
                if c == 'K' {
                    kings[0] += 1;
                }
                if c == 'k' {
                    kings[1] += 1;
                }
            } else {
                return Err(format!(
                    "unsupported FEN symbol: {c}; use B for alfil and Q for fers"
                ));
            }
        }
        if files != 8 {
            return Err("each FEN rank must contain eight squares".into());
        }
    }
    if kings != [1, 1] || pieces > 32 {
        return Err("expected exactly one king per colour and at most 32 pieces".into());
    }
    // Reject i16::MIN: negating it for Black's perspective overflows.
    let score: i16 = fields[1].parse().map_err(|_| "score must fit in i16")?;
    if score == i16::MIN {
        return Err("score must be between -32767 and 32767".into());
    }
    line.parse()
}

pub fn prepare(input: &Path, output: &Path) -> Result<u64, String> {
    let reader = BufReader::new(File::open(input).map_err(|e| e.to_string())?);
    // Refuse to overwrite data, including an output equal to the source.
    let mut writer = BufWriter::new(
        File::options()
            .write(true)
            .create_new(true)
            .open(output)
            .map_err(|e| e.to_string())?,
    );
    let result = (|| {
        let mut count = 0;
        for (i, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| e.to_string())?;
            if line.trim().is_empty() {
                continue;
            }
            let board = parse_record(&line).map_err(|e| format!("line {}: {e}", i + 1))?;
            writer
                .write_all(&board.occ.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer.write_all(&board.pcs).map_err(|e| e.to_string())?;
            writer
                .write_all(&board.score.to_le_bytes())
                .map_err(|e| e.to_string())?;
            writer
                .write_all(&[board.result, board.ksq, board.opp_ksq])
                .map_err(|e| e.to_string())?;
            writer.write_all(&board.extra).map_err(|e| e.to_string())?;
            count += 1;
        }
        writer.flush().map_err(|e| e.to_string())?;
        if count == 0 {
            return Err("dataset is empty".into());
        }
        Ok(count)
    })();
    drop(writer);
    if result.is_err() {
        std::fs::remove_file(output)
            .map_err(|e| format!("failed to remove incomplete dataset: {e}"))?;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_relative_labels_become_side_to_move_relative() {
        let board = parse_record("7k/8/8/8/8/8/8/K7 b - - 0 1 | 120 | 1.0").unwrap();
        assert_eq!(board.score, -120);
        assert_eq!(board.result, 0);
        assert_eq!(board.ksq, 7);
        assert_eq!(board.opp_ksq, 0);
    }

    #[test]
    fn rejects_malformed_data() {
        for line in [
            "8/8/8/8/8/8/8/K7 w - - 0 1 | 0 | 0.5",
            "7k/8/8/8/8/8/8/KA6 w - - 0 1 | 0 | 0.5",
            "7k/8/8/8/8/8/8/K8 w - - 0 1 | 0 | 0.5",
            "7k/8/8/8/8/8/8/K7 w K - 0 1 | 0 | 0.5",
            "7k/8/8/8/8/8/8/K7 b - - 0 1 | -32768 | 0.5",
        ] {
            assert!(parse_record(line).is_err(), "{line}");
        }
    }

    #[test]
    fn preparation_removes_incomplete_output_and_preserves_existing_files() {
        let directory = std::env::temp_dir().join(format!(
            "fsf-shatranj-data-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&directory).unwrap();
        let source = directory.join("source.txt");
        let output = directory.join("data.bf");
        std::fs::write(
            &source,
            "7k/8/8/8/8/8/8/K7 w - - 0 1 | 120 | 1.0\ninvalid record\n",
        )
        .unwrap();
        assert!(prepare(&source, &output).is_err());
        assert!(!output.exists());
        std::fs::write(&source, "7k/8/8/8/8/8/8/K7 b - - 0 1 | 120 | 1.0\n").unwrap();
        assert_eq!(prepare(&source, &output).unwrap(), 1);
        let bytes = std::fs::read(&output).unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(i16::from_le_bytes([bytes[24], bytes[25]]), -120);
        assert_eq!(&bytes[26..29], &[0, 7, 0]);
        assert!(prepare(&source, &output).is_err());
        assert_eq!(std::fs::read(&output).unwrap(), bytes);
        std::fs::remove_file(source).unwrap();
        std::fs::remove_file(output).unwrap();
        std::fs::remove_dir(directory).unwrap();
    }
}
