pub const HELP: &str = "Shatranj NNUE trainer for Fairy-Stockfish

  sf-bullet-train prepare INPUT.txt OUTPUT.bf
  sf-bullet-train train DATA.bf [OPTIONS]

Text records: Shatranj FEN | White-relative score | White-relative result
Use B/b for alfil and Q/q for fers. Results: 0.0, 0.5, 1.0.
Training input is Bullet ChessBoard data, not Stockfish/variant binpack.

Options (all take a value):
  --output DIR                 Checkpoint directory [checkpoints]
  --net-id NAME                Checkpoint prefix [shatranj]
  --batch-size N               Positions per batch [16384]
  --batches-per-superbatch N   Batches per superbatch [1024]
  --superbatches N             Number of superbatches [150]
  --start-superbatch N         Starting superbatch [1]
  --save-rate N                Save every N superbatches [10]
  --learning-rate FLOAT        Initial learning rate [0.001]
  --wdl FLOAT                  Result target proportion, 0..1 [0.0]
  --threads N                  Data preparation threads [4]
  --resume DIR                 Load compatible optimiser checkpoint

Default build uses CUDA. CPU smoke tests: --no-default-features --features cpu";

pub struct Config {
    pub data: String,
    pub output: String,
    pub net_id: String,
    pub batch_size: usize,
    pub batches: usize,
    pub superbatches: usize,
    pub start: usize,
    pub save_rate: usize,
    pub learning_rate: f32,
    pub wdl: f32,
    pub threads: usize,
    pub resume: Option<String>,
}

impl Config {
    pub fn parse(args: &[String]) -> Result<Self, String> {
        if args.len() < 2 || args[0] != "train" {
            return Err("expected train DATA.bf; see --help".into());
        }
        let mut config = Self {
            data: args[1].clone(),
            output: "checkpoints".into(),
            net_id: "shatranj".into(),
            batch_size: 16_384,
            batches: 1024,
            superbatches: 150,
            start: 1,
            save_rate: 10,
            learning_rate: 0.001,
            wdl: 0.0,
            threads: 4,
            resume: None,
        };
        let mut options = args[2..].iter();
        while let Some(key) = options.next() {
            let value = options
                .next()
                .ok_or_else(|| format!("missing value for {key}"))?;
            let int = || {
                value
                    .parse::<usize>()
                    .map_err(|_| format!("invalid integer for {key}"))
            };
            let float = || {
                value
                    .parse::<f32>()
                    .map_err(|_| format!("invalid number for {key}"))
            };
            match key.as_str() {
                "--output" => config.output = value.clone(),
                "--net-id" => config.net_id = value.clone(),
                "--batch-size" => config.batch_size = int()?,
                "--batches-per-superbatch" => config.batches = int()?,
                "--superbatches" => config.superbatches = int()?,
                "--start-superbatch" => config.start = int()?,
                "--save-rate" => config.save_rate = int()?,
                "--threads" => config.threads = int()?,
                "--learning-rate" => config.learning_rate = float()?,
                "--wdl" => config.wdl = float()?,
                "--resume" => config.resume = Some(value.clone()),
                _ => return Err(format!("unknown option: {key}")),
            }
        }
        if [
            config.batch_size,
            config.batches,
            config.superbatches,
            config.start,
            config.save_rate,
            config.threads,
        ]
        .contains(&0)
        {
            return Err("batch sizes, counts, save rate and threads must be positive".into());
        }
        if config.batch_size > 256 * 1024 * 1024 / 32 {
            return Err("batch size exceeds the Bullet loader's 256 MiB buffer".into());
        }
        if !config.learning_rate.is_finite()
            || config.learning_rate <= 0.0
            || !(0.0..=1.0).contains(&config.wdl)
        {
            return Err(
                "learning rate must be positive and finite; WDL proportion must be in [0,1]".into(),
            );
        }
        if config.net_id.is_empty()
            || !config
                .net_id
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        {
            return Err("net id must contain only letters, digits, hyphens and underscores".into());
        }
        config
            .start
            .checked_add(config.superbatches - 1)
            .ok_or("superbatch range overflows")?;
        if config.start > 1 && config.resume.is_none() {
            return Err("--start-superbatch above 1 requires --resume".into());
        }
        Ok(config)
    }
}
