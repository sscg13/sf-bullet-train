/* Attack generation is adapted from montytrain (https://github.com/official-monty/montytrain). */

use bullet::game::inputs::SparseInputType;
use bulletformat::ChessBoard;

macro_rules! init {
    (|$sq:ident, $size:literal | $($rest:tt)+) => {{
        let mut $sq = 0;
        let mut res = [{$($rest)+}; $size];
        while $sq < $size {
            res[$sq] = {$($rest)+};
            $sq += 1;
        }
        res
    }};
}

pub struct Side;
impl Side {
    pub const WHITE: usize = 0;
    pub const BLACK: usize = 1;
}

pub struct Piece;
impl Piece {
    pub const PAWN: usize = 2;
    pub const KNIGHT: usize = 3;
    pub const BISHOP: usize = 4;
    pub const ROOK: usize = 5;
    pub const QUEEN: usize = 6;
    pub const KING: usize = 7;
}

const PAWN_TYPE: usize = Piece::PAWN - 2;
const KNIGHT_TYPE: usize = Piece::KNIGHT - 2;
const BISHOP_TYPE: usize = Piece::BISHOP - 2;
const ROOK_TYPE: usize = Piece::ROOK - 2;
const QUEEN_TYPE: usize = Piece::QUEEN - 2;
const KING_TYPE: usize = Piece::KING - 2;

const NUM_VALID_TARGETS: [usize; 12] = [6, 6, 10, 10, 8, 8, 8, 8, 10, 10, 0, 0];
const TARGET_MAP: [[i32; 6]; 6] = [
    [0, 1, -1, 2, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, -1, -1],
    [0, 1, 2, 3, 4, -1],
    [-1, -1, -1, -1, -1, -1],
];

pub mod offsets {
    pub const END: usize = super::THREAT_FEATURES;
}

type ThreatOffsetTable = [[usize; 66]; 12];

struct ThreatFeatureCalculation {
    table: ThreatOffsetTable,
    total_features: usize,
}

const fn piece_id(piece_type: usize, color: usize) -> usize {
    2 * piece_type + color
}

const fn pawn_attacks_from(sq: usize, color: usize) -> u64 {
    let bit = 1u64 << sq;
    if color == Side::WHITE {
        ((bit & !File::A) << 7) | ((bit & !File::H) << 9)
    } else {
        ((bit & !File::A) >> 9) | ((bit & !File::H) >> 7)
    }
}

const fn pawn_push_from(sq: usize, color: usize) -> u64 {
    if color == Side::WHITE {
        if sq <= 55 {
            1u64 << (sq + 8)
        } else {
            0
        }
    } else if sq >= 8 {
        1u64 << (sq - 8)
    } else {
        0
    }
}

const fn pseudo_attacks(piece_type: usize, sq: usize) -> u64 {
    match piece_type {
        KNIGHT_TYPE => attacks::KNIGHT[sq],
        BISHOP_TYPE => attacks::BISHOP[sq],
        ROOK_TYPE => attacks::ROOK[sq],
        QUEEN_TYPE => attacks::QUEEN[sq],
        KING_TYPE => attacks::KING[sq],
        _ => 0,
    }
}

const fn threat_feature_calculation() -> ThreatFeatureCalculation {
    let mut table = [[0; 66]; 12];
    let mut piece_offset = 0;

    let mut color = 0;
    while color < 2 {
        let mut piece_type = 0;
        while piece_type < 6 {
            let piece = piece_id(piece_type, color);
            table[piece][65] = piece_offset;

            let mut square_offset = 0;
            let mut from = 0;
            while from < 64 {
                table[piece][from] = square_offset;

                if piece_type == PAWN_TYPE {
                    if from >= 8 && from <= 55 {
                        square_offset += (pawn_attacks_from(from, color)
                            | pawn_push_from(from, color))
                        .count_ones() as usize;
                    }
                } else {
                    square_offset += pseudo_attacks(piece_type, from).count_ones() as usize;
                }

                from += 1;
            }

            table[piece][64] = square_offset;
            piece_offset += NUM_VALID_TARGETS[piece] * square_offset;
            piece_type += 1;
        }
        color += 1;
    }

    ThreatFeatureCalculation {
        table,
        total_features: piece_offset,
    }
}

const THREAT_FEATURE_CALCULATION: ThreatFeatureCalculation = threat_feature_calculation();
const THREAT_OFFSETS: ThreatOffsetTable = THREAT_FEATURE_CALCULATION.table;
const THREAT_FEATURES: usize = THREAT_FEATURE_CALCULATION.total_features;
const _: () = assert!(THREAT_FEATURES == 60_720);

pub mod attacks {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = A << 7;

    const DIAGS: [u64; 15] = [
        0x0100_0000_0000_0000,
        0x0201_0000_0000_0000,
        0x0402_0100_0000_0000,
        0x0804_0201_0000_0000,
        0x1008_0402_0100_0000,
        0x2010_0804_0201_0000,
        0x4020_1008_0402_0100,
        0x8040_2010_0804_0201,
        0x0080_4020_1008_0402,
        0x0000_8040_2010_0804,
        0x0000_0080_4020_1008,
        0x0000_0000_8040_2010,
        0x0000_0000_0080_4020,
        0x0000_0000_0000_8040,
        0x0000_0000_0000_0080,
    ];

    pub const KNIGHT: [u64; 64] = init!(|sq, 64| {
        let n = 1 << sq;
        let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
        let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
        (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
    });

    pub const BISHOP: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        DIAGS[file + rank].swap_bytes() ^ DIAGS[7 + file - rank]
    });

    pub const ROOK: [u64; 64] = init!(|sq, 64| {
        let rank = sq / 8;
        let file = sq % 8;
        (0xFF << (rank * 8)) ^ (A << file)
    });

    pub const QUEEN: [u64; 64] = init!(|sq, 64| BISHOP[sq] | ROOK[sq]);

    pub const KING: [u64; 64] = init!(|sq, 64| {
        let mut k = 1 << sq;
        k |= (k << 8) | (k >> 8);
        k |= ((k & !A) >> 1) | ((k & !H) << 1);
        k ^ (1 << sq)
    });
}

pub struct Attacks;
impl Attacks {
    #[inline]
    pub fn pawn(sq: usize, side: usize) -> u64 {
        LOOKUP.pawn[side][sq]
    }

    #[inline]
    pub fn knight(sq: usize) -> u64 {
        LOOKUP.knight[sq]
    }

    #[inline]
    pub fn king(sq: usize) -> u64 {
        LOOKUP.king[sq]
    }

    // hyperbola quintessence
    // this gets automatically vectorised when targeting avx or better
    #[inline]
    pub fn bishop(sq: usize, occ: u64) -> u64 {
        let mask = LOOKUP.bishop[sq];

        let mut diag = occ & mask.diag;
        let mut rev1 = diag.swap_bytes();
        diag = diag.wrapping_sub(mask.bit);
        rev1 = rev1.wrapping_sub(mask.swap);
        diag ^= rev1.swap_bytes();
        diag &= mask.diag;

        let mut anti = occ & mask.anti;
        let mut rev2 = anti.swap_bytes();
        anti = anti.wrapping_sub(mask.bit);
        rev2 = rev2.wrapping_sub(mask.swap);
        anti ^= rev2.swap_bytes();
        anti &= mask.anti;

        diag | anti
    }

    // shifted lookup
    // files and ranks are mapped to 1st rank and looked up by occupancy
    #[inline]
    pub fn rook(sq: usize, occ: u64) -> u64 {
        let flip = ((occ >> (sq & 7)) & File::A).wrapping_mul(DIAG);
        let file_sq = (flip >> 57) & 0x3F;
        let files = LOOKUP.file[sq][file_sq as usize];

        let rank_sq = (occ >> RANK_SHIFT[sq]) & 0x3F;
        let ranks = LOOKUP.rank[sq][rank_sq as usize];

        ranks | files
    }

    #[inline]
    pub fn queen(sq: usize, occ: u64) -> u64 {
        Self::bishop(sq, occ) | Self::rook(sq, occ)
    }
}

struct File;
impl File {
    const A: u64 = 0x0101_0101_0101_0101;
    const H: u64 = Self::A << 7;
}

const EAST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) ^ (1 << sq) ^ WEST[sq]);
const WEST: [u64; 64] = init!(|sq, 64| (0xFF << (sq & 56)) & ((1 << sq) - 1));
const DIAG: u64 = DIAGS[7];
const DIAGS: [u64; 15] = [
    0x0100_0000_0000_0000,
    0x0201_0000_0000_0000,
    0x0402_0100_0000_0000,
    0x0804_0201_0000_0000,
    0x1008_0402_0100_0000,
    0x2010_0804_0201_0000,
    0x4020_1008_0402_0100,
    0x8040_2010_0804_0201,
    0x0080_4020_1008_0402,
    0x0000_8040_2010_0804,
    0x0000_0080_4020_1008,
    0x0000_0000_8040_2010,
    0x0000_0000_0080_4020,
    0x0000_0000_0000_8040,
    0x0000_0000_0000_0080,
];

// masks for hyperbola quintessence bishop attacks
#[derive(Clone, Copy)]
struct Mask {
    bit: u64,
    diag: u64,
    anti: u64,
    swap: u64,
}

struct Lookup {
    pawn: [[u64; 64]; 2],
    knight: [u64; 64],
    king: [u64; 64],
    bishop: [Mask; 64],
    rank: [[u64; 64]; 64],
    file: [[u64; 64]; 64],
}

static LOOKUP: Lookup = Lookup {
    pawn: PAWN,
    knight: KNIGHT,
    king: KING,
    bishop: BISHOP,
    rank: RANK,
    file: FILE,
};

const PAWN: [[u64; 64]; 2] = [
    init!(|sq, 64| (((1 << sq) & !File::A) << 7) | (((1 << sq) & !File::H) << 9)),
    init!(|sq, 64| (((1 << sq) & !File::A) >> 9) | (((1 << sq) & !File::H) >> 7)),
];

const KNIGHT: [u64; 64] = init!(|sq, 64| {
    let n = 1 << sq;
    let h1 = ((n >> 1) & 0x7f7f_7f7f_7f7f_7f7f) | ((n << 1) & 0xfefe_fefe_fefe_fefe);
    let h2 = ((n >> 2) & 0x3f3f_3f3f_3f3f_3f3f) | ((n << 2) & 0xfcfc_fcfc_fcfc_fcfc);
    (h1 << 16) | (h1 >> 16) | (h2 << 8) | (h2 >> 8)
});

const KING: [u64; 64] = init!(|sq, 64| {
    let mut k = 1 << sq;
    k |= (k << 8) | (k >> 8);
    k |= ((k & !File::A) >> 1) | ((k & !File::H) << 1);
    k ^ (1 << sq)
});

const BISHOP: [Mask; 64] = init!(|sq, 64|
    let bit = 1 << sq;
    let file = sq & 7;
    let rank = sq / 8;
    Mask {
        bit,
        diag: bit ^ DIAGS[7 + file - rank],
        anti: bit ^ DIAGS[    file + rank].swap_bytes(),
        swap: bit.swap_bytes()
    }
);

const RANK_SHIFT: [usize; 64] = init!(|sq, 64| sq - (sq & 7) + 1);

const RANK: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| {
    let file = sq & 7;
    let mask = (occ << 1) as u64;
    let east = ((EAST[file] & mask) | (1 << 63)).trailing_zeros() as usize;
    let west = ((WEST[file] & mask) | 1).leading_zeros() as usize ^ 63;
    (EAST[file] ^ EAST[east] | WEST[file] ^ WEST[west]) << (sq - file)
}));

const FILE: [[u64; 64]; 64] = init!(|sq, 64| init!(|occ, 64| (RANK[7 - sq / 8][occ]
    .wrapping_mul(DIAG)
    & File::H)
    >> (7 - (sq & 7))));

fn orient_mask(perspective: usize, ksq: usize) -> usize {
    let horizontal = if ksq % 8 < 4 { 0 } else { 7 };
    let vertical = if perspective == Side::BLACK { 56 } else { 0 };
    horizontal ^ vertical
}

fn threat_index(
    perspective: usize,
    mut attacker_color: usize,
    attacker: usize,
    mut from: usize,
    mut to: usize,
    mut attacked_color: usize,
    attacked: usize,
    ksq: usize,
) -> Option<usize> {
    let enemy = attacker_color != attacked_color;
    let orientation = orient_mask(perspective, ksq);
    from ^= orientation;
    to ^= orientation;

    if perspective == Side::BLACK {
        attacker_color ^= 1;
        attacked_color ^= 1;
    }

    let attacker_type = attacker - 2;
    let attacked_type = attacked - 2;
    let mapped_target = TARGET_MAP[attacker_type][attacked_type];

    if mapped_target < 0
        || (attacker_type == attacked_type && (enemy || attacker_type != PAWN_TYPE) && from < to)
    {
        return None;
    }

    let attacker_id = piece_id(attacker_type, attacker_color);
    let attacks = if attacker_type == PAWN_TYPE {
        pawn_attacks_from(from, attacker_color) | pawn_push_from(from, attacker_color)
    } else {
        pseudo_attacks(attacker_type, from)
    };
    let below_to = (attacks & ((1u64 << to) - 1)).count_ones() as usize;
    let target_offset =
        attacked_color * (NUM_VALID_TARGETS[attacker_id] / 2) + mapped_target as usize;
    let threat = THREAT_OFFSETS[attacker_id][65]
        + target_offset * THREAT_OFFSETS[attacker_id][64]
        + THREAT_OFFSETS[attacker_id][from]
        + below_to;

    assert!(threat < offsets::END, "{threat}");
    Some(threat)
}

fn map_bb<F: FnMut(usize)>(mut bb: u64, mut f: F) {
    while bb > 0 {
        let sq = bb.trailing_zeros() as usize;
        f(sq);
        bb &= bb - 1;
    }
}

fn map_full_threats<F: FnMut(usize)>(bbs: [u64; 8], perspective: usize, ksq: usize, mut f: F) {
    let mut piece_on = [usize::MAX; 64];
    let mut color_on = [usize::MAX; 64];
    for side in [Side::WHITE, Side::BLACK] {
        for piece in Piece::PAWN..=Piece::KING {
            map_bb(bbs[side] & bbs[piece], |sq| {
                piece_on[sq] = piece;
                color_on[sq] = side;
            });
        }
    }

    let occ = bbs[Side::WHITE] | bbs[Side::BLACK];
    let pawn_occ = bbs[Piece::PAWN];
    let color_order = if perspective == Side::WHITE {
        [Side::WHITE, Side::BLACK]
    } else {
        [Side::BLACK, Side::WHITE]
    };

    for side in color_order {
        for piece in Piece::PAWN..=Piece::KING {
            let pieces = bbs[side] & bbs[piece];

            if piece == Piece::PAWN {
                map_bb(pieces, |from| {
                    let attacks = Attacks::pawn(from, side) & occ;
                    map_bb(attacks, |to| {
                        if let Some(idx) = threat_index(
                            perspective,
                            side,
                            piece,
                            from,
                            to,
                            color_on[to],
                            piece_on[to],
                            ksq,
                        ) {
                            f(idx);
                        }
                    });

                    let push = (if side == Side::WHITE {
                        if from <= 55 {
                            1u64 << (from + 8)
                        } else {
                            0
                        }
                    } else if from >= 8 {
                        1u64 << (from - 8)
                    } else {
                        0
                    }) & pawn_occ;

                    map_bb(push, |to| {
                        if let Some(idx) = threat_index(
                            perspective,
                            side,
                            piece,
                            from,
                            to,
                            color_on[to],
                            piece_on[to],
                            ksq,
                        ) {
                            f(idx);
                        }
                    });
                });
            } else {
                map_bb(pieces, |from| {
                    let attacks = match piece {
                        Piece::KNIGHT => Attacks::knight(from),
                        Piece::BISHOP => Attacks::bishop(from, occ),
                        Piece::ROOK => Attacks::rook(from, occ),
                        Piece::QUEEN => Attacks::queen(from, occ),
                        Piece::KING => Attacks::king(from),
                        _ => unreachable!(),
                    } & occ;

                    map_bb(attacks, |to| {
                        if let Some(idx) = threat_index(
                            perspective,
                            side,
                            piece,
                            from,
                            to,
                            color_on[to],
                            piece_on[to],
                            ksq,
                        ) {
                            f(idx);
                        }
                    });
                });
            }
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct ThreatInputsBucketsMirrored;
impl ThreatInputsBucketsMirrored {
    #[rustfmt::skip]
    pub const BUCKETS: [usize; 64] = [
        28, 29, 30, 31, 31, 30, 29, 28,
        24, 25, 26, 27, 27, 26, 25, 24,
        20, 21, 22, 23, 23, 22, 21, 20,
        16, 17, 18, 19, 19, 18, 17, 16,
        12, 13, 14, 15, 15, 14, 13, 12,
        08, 09, 10, 11, 11, 10, 09, 08,
        04, 05, 06, 07, 07, 06, 05, 04,
        00, 01, 02, 03, 03, 02, 01, 00,
    ];
    pub const MK_SIZE: usize = 704;
    pub const BUCKET_COUNT: usize = 32;
    pub const FACTORISER_SIZE: usize = 768;
    pub const THREATS_SIZE: usize = offsets::END;
    pub const HALFKA_V2_SIZE: usize = 704 * Self::BUCKET_COUNT;

    pub const THREATS_MAX_ACTIVE: usize = 128; // includes PSQ
    pub const FACTORISER_MAX_ACTIVE: usize = 32;

    pub fn make_index(perspective: usize, sq: usize, pc: u8, ksq: usize) -> usize {
        let flip = 56 * perspective;
        let orientation = if ksq % 8 > 3 { 0 } else { 7 };
        let color = usize::from(pc & 8 > 0);
        let pctype = usize::from(pc & 7);
        let sf_pc: usize = 2 * pctype + color ^ perspective;
        let sf_pc = sf_pc.min(10); // no king feature on index 11
        return (sq ^ flip ^ orientation) + 64 * sf_pc + 64 * 11 * Self::BUCKETS[ksq];
    }

    /// This function should only ever be called with HalfKAv2-related indices
    pub fn derive_factorised_feature(feat: usize) -> usize {
        let mut feature = feat % 704;

        let square = feat % 64;
        let bucket = feat / 704;
        let piece = feature / 64;
        if piece == 10
            && (square % 8 <= 3 || ThreatInputsBucketsMirrored::BUCKETS[square] != bucket)
        {
            // If the feature corresponds to a king,
            // and the king is in a different bucket or in an impossible
            // position due to mirroring, it's the ntm king,
            // which is encoded differently in Chess768 inputs
            feature += 64; // effectively makes "piece" become 11
        }

        feature
    }
}
impl SparseInputType for ThreatInputsBucketsMirrored {
    type RequiredDataType = ChessBoard;

    fn num_inputs(&self) -> usize {
        Self::FACTORISER_SIZE + Self::THREATS_SIZE + Self::HALFKA_V2_SIZE
    }

    fn max_active(&self) -> usize {
        Self::THREATS_MAX_ACTIVE + Self::FACTORISER_MAX_ACTIVE
    }

    fn map_features<F: FnMut(usize, usize)>(&self, pos: &Self::RequiredDataType, mut f: F) {
        for (piece, square) in pos.into_iter() {
            let stm = Self::make_index(0 as usize, square as usize, piece, pos.our_ksq() as usize);
            let ntm = Self::make_index(1 as usize, square as usize, piece, pos.opp_ksq() as usize);
            // bucketed feature
            f(Self::FACTORISER_SIZE + stm, Self::FACTORISER_SIZE + ntm);
            // factorised feature
            f(
                Self::derive_factorised_feature(stm),
                Self::derive_factorised_feature(ntm),
            );
        }

        let mut bbs = [0; 8];
        for (pc, sq) in pos.into_iter() {
            let pt = 2 + usize::from(pc & 7);
            let c = usize::from(pc & 8 > 0);
            let bit = 1 << sq;
            bbs[c] |= bit;
            bbs[pt] |= bit;
        }

        let mut stm_count = 0;
        let mut stm_feats = [0; Self::THREATS_MAX_ACTIVE];
        map_full_threats(bbs, Side::WHITE, pos.our_ksq() as usize, |stm| {
            assert!(stm_count < Self::THREATS_MAX_ACTIVE);
            stm_feats[stm_count] = stm;
            stm_count += 1;
        });

        let mut ntm_count = 0;
        let mut ntm_feats = [0; Self::THREATS_MAX_ACTIVE];
        map_full_threats(bbs, Side::BLACK, pos.opp_ksq() as usize, |ntm| {
            assert!(ntm_count < Self::THREATS_MAX_ACTIVE);
            ntm_feats[ntm_count] = ntm;
            ntm_count += 1;
        });

        assert_eq!(stm_count, ntm_count);

        for (&stm, &ntm) in stm_feats.iter().zip(ntm_feats.iter()).take(stm_count) {
            // threat feature
            f(
                Self::FACTORISER_SIZE + Self::HALFKA_V2_SIZE + stm,
                Self::FACTORISER_SIZE + Self::HALFKA_V2_SIZE + ntm,
            );
        }
    }

    fn shorthand(&self) -> String {
        format!(
            "{}x{}+{}",
            Self::THREATS_SIZE,
            Self::MK_SIZE,
            Self::BUCKET_COUNT
        )
    }

    fn description(&self) -> String {
        "Factorised HalfKAv2 + Full_Threats inputs".to_string()
    }
}
