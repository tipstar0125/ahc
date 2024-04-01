#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use bitset_fixed::BitSet;
use proconio::{input, marker::Usize1};
use rand::prelude::*;

fn main() {
    let start = std::time::Instant::now();

    solve();

    #[allow(unused_mut, unused_assignments)]
    let mut elapsed_time = start.elapsed().as_micros() as f64 * 1e-6;
    #[cfg(feature = "local")]
    {
        eprintln!("Local Mode");
        elapsed_time *= 0.55;
    }
    eprintln!("Elapsed: {}", (elapsed_time * 1000.0) as usize);
}

fn solve() {
    // let mut rng = rand_pcg::Pcg64Mcg::new(0);
    let input = read_input();
    let state = State::new(&input);
    state.output(&input);

    eprintln!("Score: {}", state.calc_cost(&input));
}

struct State {
    col_num: Vec<Vec<usize>>,
    col_num_set: Vec<BitSet>,
    necessary_col_num: Vec<Vec<usize>>,
    prefix_col_num: Vec<Vec<usize>>,
}

impl State {
    fn new(input: &Input) -> Self {
        let mut necessary_col_num = vec![vec![0; input.N]; input.D];
        for d in 0..input.D {
            for n in 0..input.N {
                necessary_col_num[d][n] = (input.A[d][n] + input.W - 1) / input.W;
            }
        }

        let mut max_prefix_col_num = vec![0; input.N + 1];
        let mut max_col_num = vec![0; input.N];
        for n in 0..input.N {
            let mut mx = 0;
            for d in 0..input.D {
                if (input.A[d][n] + input.W - 1) / input.W > mx {
                    mx = (input.A[d][n] + input.W - 1) / input.W;
                }
            }
            max_col_num[n] = mx;
            max_prefix_col_num[n + 1] = max_prefix_col_num[n] + mx;
        }
        let mut fixed_sep_pos = input.N;
        for d in 0..input.D {
            let mut limit = input.W;
            for n in (0..input.N).rev() {
                if max_prefix_col_num[n + 1] <= limit {
                    fixed_sep_pos.chmin(n);
                    eprintln!("{} {}", d, n);
                    break;
                }
                if limit >= necessary_col_num[d][n] {
                    limit -= necessary_col_num[d][n];
                } else {
                    fixed_sep_pos.chmin(0);
                    eprintln!("{} {}", d, n);
                    break;
                }
            }
        }

        let mut col_num = vec![vec![0; input.N]; input.D];
        let mut col_num_set = vec![];
        let mut prefix_col_num = vec![vec![0; input.N + 1]; input.D];
        for d in 0..input.D {
            let mut cnt = 0;
            for n in 0..input.N {
                if fixed_sep_pos > 0 {
                    if n <= fixed_sep_pos {
                        col_num[d][n] = max_col_num[n];
                    } else {
                        col_num[d][n] = necessary_col_num[d][n];
                    }
                } else {
                    col_num[d][n] = necessary_col_num[d][n];
                }
                cnt += col_num[d][n];
            }
            let mut n = 0;
            while cnt > input.W {
                if col_num[d][n % input.N] == 1 {
                    n += 1;
                    continue;
                }
                col_num[d][n % input.N] -= 1;
                n += 1;
                cnt -= 1;
            }
            while cnt < input.W {
                col_num[d][input.N - 1] += 1;
                cnt += 1;
            }
            assert!(cnt == input.W);
            let mut set = BitSet::new(input.W);
            for n in 0..input.N {
                prefix_col_num[d][n + 1] = prefix_col_num[d][n] + col_num[d][n];
                if n + 1 != input.N {
                    set.set(prefix_col_num[d][n + 1], true);
                }
            }
            col_num_set.push(set);
        }

        State {
            col_num,
            col_num_set,
            necessary_col_num,
            prefix_col_num,
        }
    }
    fn calc_cost(&self, input: &Input) -> usize {
        let mut ret = 1;
        for d in 0..input.D {
            for n in 0..input.N {
                if self.col_num[d][n] < self.necessary_col_num[d][n] {
                    ret += (input.A[d][n] - self.col_num[d][n] * input.W) * 100;
                }
            }
        }
        for d in 1..input.D {
            let same = &self.col_num_set[d - 1] & &self.col_num_set[d];
            ret += (input.N - 1 - same.count_ones() as usize) * 2 * input.W;
        }
        ret
    }
    fn output(&self, input: &Input) {
        for d in 0..input.D {
            for n in 0..input.N {
                let row1 = 0;
                let col1 = self.prefix_col_num[d][n];
                let row2 = input.W;
                let col2 = self.prefix_col_num[d][n + 1];
                println!("{} {} {} {}", row1, col1, row2, col2);
            }
        }
    }
}

struct Input {
    W: usize,
    D: usize,
    N: usize,
    A: Vec<Vec<usize>>,
}

fn read_input() -> Input {
    input! {
        W: usize,
        D: usize,
        N: usize,
        A: [[usize; N]; D]
    }

    Input { W, D, N, A }
}

pub trait ChangeMinMax {
    fn chmin(&mut self, x: Self) -> bool;
    fn chmax(&mut self, x: Self) -> bool;
}

impl<T: PartialOrd> ChangeMinMax for T {
    fn chmin(&mut self, x: Self) -> bool {
        *self > x && {
            *self = x;
            true
        }
    }
    fn chmax(&mut self, x: Self) -> bool {
        *self < x && {
            *self = x;
            true
        }
    }
}

#[derive(Debug, Clone)]
struct TimeKeeper {
    start_time: std::time::Instant,
    time_threshold: f64,
}

impl TimeKeeper {
    fn new(time_threshold: f64) -> Self {
        TimeKeeper {
            start_time: std::time::Instant::now(),
            time_threshold,
        }
    }
    #[inline]
    fn isTimeOver(&self) -> bool {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55 >= self.time_threshold
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time >= self.time_threshold
        }
    }
    #[inline]
    fn get_time(&self) -> f64 {
        let elapsed_time = self.start_time.elapsed().as_nanos() as f64 * 1e-9;
        #[cfg(feature = "local")]
        {
            elapsed_time * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            elapsed_time
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Coord {
    row: usize,
    col: usize,
}

impl Coord {
    pub fn new(row: usize, col: usize) -> Self {
        Self { row, col }
    }
    pub fn in_map(&self, height: usize, width: usize) -> bool {
        self.row < height && self.col < width
    }
    pub fn to_index(&self, width: usize) -> CoordIndex {
        CoordIndex(self.row * width + self.col)
    }
}

impl std::ops::Add<CoordDiff> for Coord {
    type Output = Coord;
    fn add(self, rhs: CoordDiff) -> Self::Output {
        Coord::new(
            self.row.wrapping_add_signed(rhs.dr),
            self.col.wrapping_add_signed(rhs.dc),
        )
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoordDiff {
    dr: isize,
    dc: isize,
}

impl CoordDiff {
    pub const fn new(dr: isize, dc: isize) -> Self {
        Self { dr, dc }
    }
}

pub const ADJ: [CoordDiff; 4] = [
    CoordDiff::new(1, 0),
    CoordDiff::new(!0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(0, !0),
];

pub struct CoordIndex(pub usize);

impl CoordIndex {
    pub fn new(index: usize) -> Self {
        Self(index)
    }
    pub fn to_coord(&self, width: usize) -> Coord {
        Coord {
            row: self.0 / width,
            col: self.0 % width,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DynamicMap2d<T> {
    pub size: usize,
    map: Vec<T>,
}

impl<T> DynamicMap2d<T> {
    pub fn new(map: Vec<T>, size: usize) -> Self {
        assert_eq!(size * size, map.len());
        Self { size, map }
    }
}

impl<T: Clone> DynamicMap2d<T> {
    pub fn new_with(v: T, size: usize) -> Self {
        let map = vec![v; size * size];
        Self::new(map, size)
    }
}

impl<T> std::ops::Index<Coord> for DynamicMap2d<T> {
    type Output = T;

    #[inline]
    fn index(&self, coordinate: Coord) -> &Self::Output {
        &self[coordinate.to_index(self.size)]
    }
}

impl<T> std::ops::IndexMut<Coord> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, coordinate: Coord) -> &mut Self::Output {
        let size = self.size;
        &mut self[coordinate.to_index(size)]
    }
}

impl<T> std::ops::Index<CoordIndex> for DynamicMap2d<T> {
    type Output = T;

    fn index(&self, index: CoordIndex) -> &Self::Output {
        unsafe { self.map.get_unchecked(index.0) }
    }
}

impl<T> std::ops::IndexMut<CoordIndex> for DynamicMap2d<T> {
    #[inline]
    fn index_mut(&mut self, index: CoordIndex) -> &mut Self::Output {
        unsafe { self.map.get_unchecked_mut(index.0) }
    }
}
