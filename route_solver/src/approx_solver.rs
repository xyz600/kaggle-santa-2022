use crate::snake::create_snake_shape;
use crate::tsp;
use lib::array_solution::ArraySolution;
use lib::distance::DistanceFunction;
use lib::evaluate::evaluate;
use proconio::input;
use proconio::source::auto::AutoSource;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::str::FromStr;

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
struct Cell {
    x: i64,
    y: i64,
    r: i64,
    g: i64,
    b: i64,
}

struct ManipulationDistanceFunction {
    cell_list: Vec<Cell>,
}

impl ManipulationDistanceFunction {
    fn load(filepath: &PathBuf) -> ManipulationDistanceFunction {
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

        ManipulationDistanceFunction { cell_list }
    }
}

impl DistanceFunction for ManipulationDistanceFunction {
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

        let dist = if dx + dy < 10 {
            ((dx + dy) as f64).sqrt()
        } else {
            (dx + dy) as f64
        };

        (dist * 10000.0 * 255.0) as i64 + (dr + dg + db)
    }

    fn dimension(&self) -> u32 {
        self.cell_list.len() as u32
    }

    fn name(&self) -> String {
        "santa_2022_approx_1".to_string()
    }
}

fn reconstruct_route(route_str: Vec<char>) -> Vec<u32> {
    let mut index_table = HashMap::<(i32, i32), u32>::new();
    {
        let mut index = 0u32;
        for y in (-128..=128).rev() {
            for x in -128..=128 {
                index_table.insert((y, x), index);
                index += 1;
            }
        }
    }

    let mut y = 0;
    let mut x = 0;
    let mut solution_array = vec![index_table[&(y, x)]];

    for ch in route_str {
        if ch == 'U' {
            y += 1;
        } else if ch == 'R' {
            x += 1;
        } else if ch == 'D' {
            y -= 1;
        } else if ch == 'L' {
            x -= 1;
        } else {
            unreachable!()
        }
        let index = index_table[&(y, x)];
        solution_array.push(index);
    }
    solution_array
}

pub fn solve() {
    let input_filepath = PathBuf::from_str("data/image.csv").unwrap();
    let distance = ManipulationDistanceFunction::load(&input_filepath);

    let route_str = create_snake_shape();
    let route = reconstruct_route(route_str);
    let init_solution = ArraySolution::from_array(route);
    init_solution.save(&PathBuf::from_str("solution_snake.tsp").unwrap());
    eprintln!(
        "eval = {}",
        evaluate(&distance, &init_solution) as f64 / (255.0 * 10000.0)
    );

    tsp::solve_tsp(&distance, init_solution);
}
