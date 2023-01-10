use lib::distance::DistanceFunction;
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
    x: i64,
    y: i64,
    r: i64,
    g: i64,
    b: i64,
}

pub struct OneStepDistanceFunction {
    cell_list: Vec<Cell>,
    is_64x64_vertical: bool,
}

impl OneStepDistanceFunction {
    pub fn load(filepath: &PathBuf, is_64x64_vertical: bool) -> OneStepDistanceFunction {
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

        let cell_list = data
            .iter()
            .map(|(x, y, r, g, b)| {
                Cell {
                    x: *x,
                    y: *y,
                    r: (*r * 255.0).round() as i64 * 3 * 10000, // * 3 means cost scaling, 10000 means fixed point
                    g: (*g * 255.0).round() as i64 * 3 * 10000,
                    b: (*b * 255.0).round() as i64 * 3 * 10000,
                }
            })
            .collect::<Vec<_>>();

        OneStepDistanceFunction {
            cell_list,
            is_64x64_vertical,
        }
    }
}

impl DistanceFunction for OneStepDistanceFunction {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        // 1歩で移動できる範囲で歩いて行って全ての行動が完了できると仮定する。
        // 1歩で移動できる場合の距離は、\sqrt{|x| + |y|} で下から抑えられるので、まずはこの距離関数でいい感じのルートが作れる
        // 実際には \sqrt{|x| + |y|} だと遠い距離に対して見積もりが甘いので、後日何とかする
        let c1 = &self.cell_list[id1 as usize];
        let c2 = &self.cell_list[id2 as usize];
        let dx = (c1.x - c2.x).abs();
        let dy = (c1.y - c2.y).abs();
        let dr = (c1.r - c2.r).abs();
        let dg = (c1.g - c2.g).abs();
        let db = (c1.b - c2.b).abs();

        // 64x64 が担当する方向は 1つしか動かせない

        if self.is_64x64_vertical {
            let x_iter = (dx / 7) as f64;
            let x_rest = (dx % 7) as f64;

            if (dx + 6) / 7 <= dy {
                // (x-full + dy) + (rest + dy) + dy
                let dist =
                    8.0f64.sqrt() * x_iter + (x_rest + 1.0).sqrt() + (dy as f64) - x_iter - 1.0;
                (dist * 10000.0 * 255.0) as i64 + (dr + dg + db) * dy
            } else {
                // (x-full + dy) + x-full + x-rest
                let dist = 8.0f64.sqrt() * (dy as f64)
                    + 7.0f64.sqrt() * (x_iter - dy as f64)
                    + x_rest.sqrt();
                (dist * 10000.0 * 255.0) as i64 + (dr + dg + db) * (dx + 6) / 7
            }
        } else {
            let y_iter = (dy / 7) as f64;
            let y_rest = (dy % 7) as f64;

            if (dy + 6) / 7 <= dx {
                let dist =
                    8.0f64.sqrt() * y_iter + (y_rest + 1.0).sqrt() + (dx as f64) - y_iter - 1.0;
                (dist * 10000.0 * 255.0) as i64 + (dr + dg + db) * dx
            } else {
                let dist = 8.0f64.sqrt() * (dx as f64)
                    + 7.0f64.sqrt() * (y_iter - dx as f64)
                    + y_rest.sqrt();
                (dist * 10000.0 * 255.0) as i64 + (dr + dg + db) * (dy + 6) / 7
            }
        }
    }

    fn dimension(&self) -> u32 {
        self.cell_list.len() as u32
    }

    fn name(&self) -> String {
        "santa_2022_approx_1".to_string()
    }
}
