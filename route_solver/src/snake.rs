pub enum PosType {
    UL,
    UR,
    DR,
    DL,
}

fn rev(mut v: Vec<char>) -> Vec<char> {
    v.reverse();
    v
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
                PosType::UL => seq_cmd!(size/2; (UR, UL)),
                PosType::UR => unreachable!(),
                PosType::DR => unreachable!(),
                PosType::DL => unreachable!(),
            },
            PosType::UR => match end {
                PosType::UL => unreachable!(),
                PosType::UR => unreachable!(),
                PosType::DR => unreachable!(),
                PosType::DL => unreachable!(),
            },
            PosType::DR => match end {
                PosType::UL => unreachable!(),
                PosType::UR => unreachable!(),
                PosType::DR => unreachable!(),
                PosType::DL => unreachable!(),
            },
            PosType::DL => match end {
                PosType::UL => unreachable!(),
                PosType::UR => unreachable!(),
                PosType::DR => unreachable!(),
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

pub fn create_snake_shape() -> String {
    "".to_string()
}
