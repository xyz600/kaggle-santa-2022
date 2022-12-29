use crate::{distance::DistanceFunction, solution::Solution};

pub fn evaluate(distance: &impl DistanceFunction, solution: &impl Solution) -> i64 {
    let mut sum = 0;
    let mut id = 0;
    for _iter in 0..distance.dimension() {
        let next = solution.next(id);
        sum += distance.distance(id, next);
        id = next;
    }
    sum
}
