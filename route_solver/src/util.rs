use proconio::input;
use proconio::source::auto::AutoSource;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

macro_rules! input_fromfile {
    (path: $path:expr, $($t:tt)+) => {
        fn read_all(filepath: &PathBuf) -> String {
            let mut f = File::open(filepath).expect("file not found");
            let mut contents = String::new();

            f.read_to_string(&mut contents)
                .expect("something went wrong reading the file");

            contents
        }
        let contents = read_all($path);
        let source = AutoSource::from(contents.as_str());

        input! {
            from source,
            $($t)*
        }
    };
}

#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub x: i64,
    pub y: i64,
    pub r: i64,
    pub g: i64,
    pub b: i64,
}

pub fn load_image(filepath: &PathBuf) -> Vec<Cell> {
    const SIZE: usize = 257;
    const LEN: usize = SIZE * SIZE;

    input_fromfile! {
        path: filepath,
        _1: String,
        _2: String,
        _3: String,
        _4: String,
        _5: String,
        data: [(i64, i64, f64, f64, f64); LEN],
    };
    data.iter()
        .map(|(x, y, r, g, b)| Cell {
            x: *x,
            y: *y,
            r: (*r * 255.0).round() as i64,
            g: (*g * 255.0).round() as i64,
            b: (*b * 255.0).round() as i64,
        })
        .collect::<Vec<_>>()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coord {
    y: i16,
    x: i16,
}

impl Coord {
    pub fn new(y: i16, x: i16) -> Coord {
        Coord { y, x }
    }
    pub fn encode(&self, bit_width: i16) -> u32 {
        let offset = 1 << bit_width;
        let ey = (self.y + offset / 2) as u32 - 1;
        let ex = (self.x + offset / 2) as u32 - 1;
        ey * offset as u32 + ex
    }

    pub fn decode(value: u32, bit_width: i16) -> Coord {
        let offset = (1 << bit_width) as i16;
        let y = (value / offset as u32) as i16 - (offset / 2 - 1);
        let x = (value % offset as u32) as i16 - (offset / 2 - 1);
        Coord { y, x }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direction {
    Up,
    Left,
    Right,
    Down,
    None,
}

#[derive(Clone, Copy, Debug)]
pub struct DiffPose {
    pub diff_list: [Direction; 8],
}

#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub arm_list: [Coord; 8],
}

impl Pose {
    pub fn new() -> Pose {
        Pose {
            arm_list: [
                Coord::new(0, 64),
                Coord::new(0, -32),
                Coord::new(0, -16),
                Coord::new(0, -8),
                Coord::new(0, -4),
                Coord::new(0, -2),
                Coord::new(0, -1),
                Coord::new(0, -1),
            ],
        }
    }

    pub fn apply(&mut self, diff: &DiffPose) {
        for i in 0..8 {
            match diff.diff_list[i] {
                Direction::Up => self.arm_list[i].y += 1,
                Direction::Left => self.arm_list[i].x -= 1,
                Direction::Right => self.arm_list[i].x += 1,
                Direction::Down => self.arm_list[i].y -= 1,
                Direction::None => {}
            }
        }
    }

    pub fn encode(&self) -> u128 {
        let mut ret = 0u128;
        let bit_width_list = [9, 8, 7, 6, 5, 4, 3, 3];
        for i in 0..8 {
            ret <<= 2 * bit_width_list[i];
            ret += self.arm_list[i].encode(bit_width_list[i]) as u128;
        }
        ret
    }

    pub fn decode(mut value: u128) -> Pose {
        let mut arm_list = [Coord::new(0, 0); 8];
        let bit_width_list = [9, 8, 7, 6, 5, 4, 3, 3];

        for i in (0..8).rev() {
            let mask = (1 << (bit_width_list[i] * 2)) - 1;
            let coord_value = (value & mask) as u32;
            arm_list[i] = Coord::decode(coord_value, bit_width_list[i]);
            value >>= bit_width_list[i] * 2;
        }
        Pose { arm_list }
    }

    pub fn coord(&self) -> (i16, i16) {
        let mut y = 0;
        let mut x = 0;
        for i in 0..8 {
            y += self.arm_list[i].y;
            x += self.arm_list[i].x;
        }
        (y, x)
    }

    pub fn diff(&self, to: &Pose) -> DiffPose {
        let mut diff_list = [Direction::None; 8];
        for i in 0..8 {
            assert!(self.arm_list[i].x.abs_diff(to.arm_list[i].x) <= 1);
            assert!(self.arm_list[i].y.abs_diff(to.arm_list[i].y) <= 1);

            let cy = self.arm_list[i].y;
            let cx = self.arm_list[i].x;
            let ny = to.arm_list[i].y;
            let nx = to.arm_list[i].x;

            diff_list[i] = if cy + 1 == ny && cx == nx {
                Direction::Up
            } else if cy - 1 == ny && cx == nx {
                Direction::Down
            } else if cx + 1 == nx && cy == ny {
                Direction::Right
            } else if cx - 1 == nx && cy == ny {
                Direction::Left
            } else if cx == nx && cy == ny {
                Direction::None
            } else {
                unreachable!()
            };
        }
        DiffPose { diff_list }
    }

    pub fn pos_diff(&self, to: &Pose) -> i64 {
        let mut ret = 0;
        for i in 0..8 {
            let dy = self.arm_list[i].y - to.arm_list[i].y;
            let dx = self.arm_list[i].x - to.arm_list[i].x;
            ret += dy.abs() + dx.abs();
        }
        ret as i64
    }

    pub fn next_diff_list(&self) -> Vec<DiffPose> {
        fn get_max_value(depth: usize) -> i16 {
            let ret = [64, 32, 16, 8, 4, 2, 1, 1];
            ret[depth]
        }

        fn inner(
            pos_list: &[Coord; 8],
            depth: usize,
            ret: &mut Vec<DiffPose>,
            stack: &mut [Direction; 8],
        ) {
            if depth == 8 {
                // 全ての diff を積み終わった

                let mut changed = false;
                for i in 0..8 {
                    if stack[i] != Direction::None {
                        changed = true;
                        break;
                    }
                }
                // 全て None を避ける
                if changed {
                    ret.push(DiffPose {
                        diff_list: stack.clone(),
                    })
                }
            } else {
                let max_value = get_max_value(depth);
                // stack に積まれた方向の分も加味
                let y = pos_list[depth].y;
                let x = pos_list[depth].x;

                let candidate_dirs = if x == -max_value && y == -max_value {
                    [Direction::Up, Direction::None, Direction::Right]
                } else if x == max_value && y == -max_value {
                    [Direction::Up, Direction::None, Direction::Left]
                } else if x == -max_value && y == max_value {
                    [Direction::Down, Direction::None, Direction::Right]
                } else if x == max_value && y == max_value {
                    [Direction::Down, Direction::None, Direction::Left]
                } else if x.abs() == max_value {
                    [Direction::Up, Direction::Down, Direction::None]
                } else if y.abs() == max_value {
                    [Direction::Left, Direction::Right, Direction::None]
                } else {
                    unreachable!()
                };

                for dir in candidate_dirs {
                    stack[depth] = dir;
                    inner(pos_list, depth + 1, ret, stack);
                }
            }
        }

        let mut ret = vec![];
        let mut stack = [Direction::None; 8];
        inner(&self.arm_list, 0, &mut ret, &mut stack);
        ret
    }

    pub fn to_string(&self) -> String {
        let mut vs = vec![];
        for i in 0..8 {
            vs.push(format!("{} {}", self.arm_list[i].x, self.arm_list[i].y));
        }
        vs.join(";") + "\n"
    }
}

#[cfg(test)]
mod tests {
    use crate::{Coord, Pose};

    #[test]
    fn test_encdec() {
        let pose = Pose {
            arm_list: [
                Coord::new(-128, 128),
                Coord::new(-64, 64),
                Coord::new(32, -32),
                Coord::new(16, 16),
                Coord::new(0, 0),
                Coord::new(4, 2),
                Coord::new(2, 1),
                Coord::new(1, 2),
            ],
        };
        let reconstructed = Pose::decode(pose.encode());
        for i in 0..8 {
            assert_eq!(reconstructed.arm_list[i].y, pose.arm_list[i].y);
        }
    }
}
