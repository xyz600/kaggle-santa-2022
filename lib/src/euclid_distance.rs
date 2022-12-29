use crate::distance::DistanceFunction;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

struct Point {
    y: i64,
    x: i64,
}

impl Point {
    pub fn new(y: i64, x: i64) -> Point {
        Point { y, x }
    }

    pub fn distance(&self, other: &Point) -> i64 {
        let dy = self.y.abs_diff(other.y) as i64;
        let dx = self.x.abs_diff(other.x) as i64;
        ((dy * dy + dx * dx) as f64).sqrt().ceil() as i64
    }
}

pub struct EuclidDistance {
    point_list: Vec<Point>,
    name: String,
}

enum TSPLibFormatCode {
    Config,
    Coordinate,
}

impl EuclidDistance {
    pub fn load_tsplib(filepath: &PathBuf) -> EuclidDistance {
        let f = File::open(filepath).unwrap();
        let reader = BufReader::new(f);
        let name = filepath.file_name().unwrap().to_str().unwrap().to_string();

        let mut dimension = std::u32::MAX;
        let mut point_list = vec![];
        let mut mode = TSPLibFormatCode::Config;

        for line in reader.lines() {
            let line = line.unwrap();
            if line.contains("EOF") {
                break;
            }

            match mode {
                TSPLibFormatCode::Config => {
                    if line.contains("DIMENSION") {
                        let dim_token = line.split(": ").collect::<Vec<_>>()[1];
                        dimension = dim_token.parse::<u32>().unwrap();
                    } else if line.contains("NODE_COORD_SECTION") {
                        mode = TSPLibFormatCode::Coordinate;
                    }
                }
                TSPLibFormatCode::Coordinate => {
                    let num_token_list = line
                        .split(" ")
                        .map(|v| v.parse::<i64>().unwrap())
                        .collect::<Vec<_>>();
                    let y = num_token_list[1];
                    let x = num_token_list[2];
                    point_list.push(Point::new(y, x));
                }
            }
        }
        assert_eq!(dimension as usize, point_list.len());
        EuclidDistance { point_list, name }
    }
}

impl DistanceFunction for EuclidDistance {
    fn distance(&self, id1: u32, id2: u32) -> i64 {
        self.point_list[id1 as usize].distance(&self.point_list[id2 as usize])
    }

    fn dimension(&self) -> u32 {
        self.point_list.len() as u32
    }

    fn name(&self) -> String {
        self.name.to_string()
    }
}
