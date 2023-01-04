pub enum PosType {
    UL,
    UR,
    DR,
    DL,
}

fn rev_char(ch: char) -> char {
    if ch == 'U' {
        'D'
    } else if ch == 'D' {
        'U'
    } else if ch == 'L' {
        'R'
    } else if ch == 'R' {
        'L'
    } else {
        unreachable!()
    }
}

fn rev(v: Vec<char>) -> Vec<char> {
    v.into_iter().rev().map(rev_char).collect()
}

macro_rules! to_vec {
    ($path:expr) => {
        ($path).chars().collect()
    };
}

macro_rules! seq_cmd {
    ($size:expr; $t:tt) => {
        seq_cmd_inner!($size; $t)
    };
    ($size:expr; $t:tt $($rest:tt)+) => [{
        let mut front = seq_cmd_inner!($size; $t);
        let rest = seq_cmd![$size; $($rest)+];
        for ch in rest.iter() {
            front.push(*ch);
        }
        front
    }];
}

macro_rules! pos {
    (UL) => {
        PosType::UL
    };
    (UR) => {
        PosType::UR
    };
    (DL) => {
        PosType::DL
    };
    (DR) => {
        PosType::DR
    };
}

macro_rules! seq_cmd_inner {
    ($size:expr; ($t1:tt, $t2:tt)) => {{
        let begin = pos!($t1);
        let end = pos!($t2);
        inner_2nx2n($size, begin, end)
    }};
    ($size:expr; $t:tt) => {
        ($t).chars().collect::<Vec<char>>()
    };
}

fn inner_2x2(begin: PosType, end: PosType) -> Vec<char> {
    match begin {
        PosType::UL => match end {
            PosType::UL => unreachable!(),
            PosType::UR => to_vec!("DRU"),
            PosType::DR => unreachable!(),
            PosType::DL => to_vec!("RDL"),
        },
        PosType::UR => match end {
            PosType::UL => to_vec!("DLU"),
            PosType::UR => unreachable!(),
            PosType::DR => to_vec!("LDR"),
            PosType::DL => unreachable!(),
        },
        PosType::DR => match end {
            PosType::UL => unreachable!(),
            PosType::UR => to_vec!("LUR"),
            PosType::DR => unreachable!(),
            PosType::DL => to_vec!("ULD"),
        },
        PosType::DL => match end {
            PosType::UL => to_vec!("RUL"),
            PosType::UR => unreachable!(),
            PosType::DR => to_vec!("URD"),
            PosType::DL => unreachable!(),
        },
    }
}

pub fn inner_2nx2n(size: i64, begin: PosType, end: PosType) -> Vec<char> {
    if size == 2 {
        inner_2x2(begin, end)
    } else {
        match begin {
            PosType::UL => match end {
                PosType::UL => unreachable!(),
                PosType::UR => seq_cmd![size/2; (UL, DL) "D" (UL, UR) "R" (UL, UR) "U" (DR, UR)],
                PosType::DR => unreachable!(),
                PosType::DL => rev(inner_2nx2n(size, end, begin)),
            },
            PosType::UR => match end {
                PosType::UL => rev(inner_2nx2n(size, end, begin)),
                PosType::UR => unreachable!(),
                PosType::DR => seq_cmd![size/2; (UR, UL) "L" (UR, DR) "D" (UR, DR) "R" (DL, DR)],
                PosType::DL => unreachable!(),
            },
            PosType::DR => match end {
                PosType::UL => unreachable!(),
                PosType::UR => rev(inner_2nx2n(size, end, begin)),
                PosType::DR => unreachable!(),
                PosType::DL => seq_cmd![size/2; (DR, UR) "U" (DR, DL) "L" (DR, DL) "D" (UL, DL)],
            },
            PosType::DL => match end {
                PosType::UL => seq_cmd![size/2; (DL, DR) "R" (DL, UL) "U" (DL, UL) "L" (UR, UL)],
                PosType::UR => unreachable!(),
                PosType::DR => rev(inner_2nx2n(size, end, begin)),
                PosType::DL => unreachable!(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::inner_2nx2n;
    use super::PosType;
    #[test]
    fn test_macro1() {
        let ret = seq_cmd![2; (UR, UL)];
        assert_eq!(ret, vec!['D', 'L', 'U']);
    }

    #[test]
    fn test_macro2() {
        let ret = seq_cmd![2; "DLU"];
        assert_eq!(ret, vec!['D', 'L', 'U']);
    }

    #[test]
    fn test_macro3() {
        let ret = seq_cmd![2; "DLU"(DL, DR)];
        assert_eq!(ret, vec!['D', 'L', 'U', 'U', 'R', 'D']);
    }

    #[test]
    fn test_macro4() {
        let ret = seq_cmd![2; "DLU"(DL, DR)(DL, DR)];
        assert_eq!(ret, vec!['D', 'L', 'U', 'U', 'R', 'D', 'U', 'R', 'D']);
    }
}

pub fn create_snake_shape() -> Vec<char> {
    let size = 128;

    let mut seq = vec![
        // U x 128 + L
        vec!['U'; size],
        vec!['L'],
        // (UR, DR)
        inner_2nx2n(size as i64, PosType::UR, PosType::DR),
        // D + L x 127 + D
        vec!['D'],
        vec!['L'; size - 1],
        vec!['D'],
        // (UL, UR)
        inner_2nx2n(size as i64, PosType::UL, PosType::UR),
        // R + D x 127 + R
        vec!['R'],
        vec!['D'; size - 1],
        vec!['R'],
        // (DL, UL)
        inner_2nx2n(size as i64, PosType::DL, PosType::UL),
        // U + R x 127 + U
        vec!['U'],
        vec!['R'; size - 1],
        vec!['U'],
        // (DR, DL)
        inner_2nx2n(size as i64, PosType::DR, PosType::DL),
        // D + L (1周しようと思うと必要だけど、今は不要)
        // vec!['D', 'L'],
    ];

    let mut ret = vec![];
    for subsec in seq.iter_mut() {
        ret.append(subsec)
    }

    ret
}
