#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::collections::BTreeSet;

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
    let mut rng = rand_pcg::Pcg64Mcg::new(0);
    let input = read_input();
    let state1 = State1::new(&input);
    let state2 = State2::new(&input, &mut rng);
    let out1 = state1.output(&input, false);
    let out2 = state2.output(&input, false);
    let score1 = compute_score_details(&input, &out1);
    let score2 = compute_score_details(&input, &out2);
    if score1 < score2 {
        state1.output(&input, true);
        eprintln!("Score: {}", score1);
        eprintln!("State: {}", 1);
    } else {
        state2.output(&input, true);
        eprintln!("Score: {}", score2);
        eprintln!("State: {}", 2);
    }
}

fn calc_col(x: usize, w: usize) -> usize {
    (x + w - 1) / w
}

fn compute_score_details(input: &Input, out: &[Vec<(usize, usize, usize, usize)>]) -> i64 {
    let mut score = 0;
    let mut change: Vec<(usize, usize, usize, usize, bool)> = vec![];
    let mut hs = BTreeSet::new();
    let mut vs = BTreeSet::new();
    for d in 0..out.len() {
        for p in 0..input.N {
            for q in 0..p {
                if out[d][p].2.min(out[d][q].2) > out[d][p].0.max(out[d][q].0)
                    && out[d][p].3.min(out[d][q].3) > out[d][p].1.max(out[d][q].1)
                {
                    return 0;
                }
            }
        }
        let mut hs2 = BTreeSet::new();
        let mut vs2 = BTreeSet::new();
        for k in 0..input.N {
            let (i0, j0, i1, j1) = out[d][k];
            let b = (i1 - i0) * (j1 - j0);
            if input.A[d][k] > b {
                score += 100 * (input.A[d][k] - b) as i64;
            }
            for j in j0..j1 {
                if i0 > 0 {
                    hs2.insert((i0, j));
                }
                if i1 < input.W {
                    hs2.insert((i1, j));
                }
            }
            for i in i0..i1 {
                if j0 > 0 {
                    vs2.insert((j0, i));
                }
                if j1 < input.W {
                    vs2.insert((j1, i));
                }
            }
        }
        if d > 0 {
            for &(i, j) in &hs {
                if !hs2.contains(&(i, j)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (i, change[change.len() - 1].1, i, j, false)
                        {
                            change.last_mut().unwrap().3 += 1;
                        } else {
                            change.push((i, j, i, j + 1, false));
                        }
                    }
                }
            }
            for &(j, i) in &vs {
                if !vs2.contains(&(j, i)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (change[change.len() - 1].0, j, i, j, false)
                        {
                            change.last_mut().unwrap().2 += 1;
                        } else {
                            change.push((i, j, i + 1, j, false));
                        }
                    }
                }
            }
            for &(i, j) in &hs2 {
                if !hs.contains(&(i, j)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (i, change[change.len() - 1].1, i, j, true)
                        {
                            change.last_mut().unwrap().3 += 1;
                        } else {
                            change.push((i, j, i, j + 1, true));
                        }
                    }
                }
            }
            for &(j, i) in &vs2 {
                if !vs.contains(&(j, i)) {
                    score += 1;
                    if d + 1 == out.len() {
                        if !change.is_empty()
                            && change[change.len() - 1]
                                == (change[change.len() - 1].0, j, i, j, true)
                        {
                            change.last_mut().unwrap().2 += 1;
                        } else {
                            change.push((i, j, i + 1, j, true));
                        }
                    }
                }
            }
        }
        hs = hs2;
        vs = vs2;
    }
    score + 1
}

struct State1 {
    col_num: Vec<Vec<usize>>,
    col_num_set: Vec<BitSet>,
    necessary_col_num: Vec<Vec<usize>>,
    prefix_col_num: Vec<Vec<usize>>,
}

impl State1 {
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
                    // eprintln!("{} {}", d, n);
                    break;
                }
                if limit >= necessary_col_num[d][n] {
                    limit -= necessary_col_num[d][n];
                } else {
                    fixed_sep_pos.chmin(0);
                    // eprintln!("{} {}", d, n);
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

        State1 {
            col_num,
            col_num_set,
            necessary_col_num,
            prefix_col_num,
        }
    }
    fn output(&self, input: &Input, flag: bool) -> Vec<Vec<(usize, usize, usize, usize)>> {
        let mut ret = vec![];
        for d in 0..input.D {
            let mut out = vec![];
            for n in 0..input.N {
                let row1 = 0;
                let col1 = self.prefix_col_num[d][n];
                let row2 = input.W;
                let col2 = self.prefix_col_num[d][n + 1];
                out.push((row1, col1, row2, col2));
                if flag {
                    println!("{} {} {} {}", row1, col1, row2, col2);
                }
            }
            ret.push(out);
        }
        ret
    }
}

struct State2 {
    rows1: Vec<Vec<usize>>,
    rows2: Vec<Vec<usize>>,
    col_num_list1: Vec<Vec<usize>>,
    col_num_list2: Vec<Vec<usize>>,
    prefix_sum1: Vec<Vec<usize>>,
    prefix_sum2: Vec<Vec<usize>>,
}

impl State2 {
    fn new(input: &Input, rng: &mut rand_pcg::Pcg64Mcg) -> Self {
        let half = input.W / 2;
        let mut used_list = vec![];
        for d in 0..input.D {
            let mut sum = 0;
            let mut used = vec![false; input.N];
            for _ in 0..input.N * 10 {
                let idx = rng.gen_range(0..input.N);
                if used[idx] {
                    continue;
                }
                let col = calc_col(input.A[d][idx], half);
                if sum + col <= input.W {
                    sum += col;
                    used[idx] = true;
                }
            }

            let mut best_used = used.clone();
            let mut best_sum = sum;
            let mut col1 = 0;
            let mut col2 = 0;
            for n in 0..input.N {
                if used[n] {
                    col1 += calc_col(input.A[d][n], half);
                } else {
                    col2 += calc_col(input.A[d][n], half);
                }
            }

            for _ in 0..input.N * 10000 {
                if col1 <= input.W && col2 <= input.W {
                    best_used = used.clone();
                    break;
                }
                let used_idx = {
                    let mut idx;
                    loop {
                        idx = rng.gen_range(0..input.N);
                        if used[idx] {
                            break;
                        }
                    }
                    idx
                };
                let not_used_idx = {
                    let mut idx;
                    loop {
                        idx = rng.gen_range(0..input.N);
                        if !used[idx] {
                            break;
                        }
                    }
                    idx
                };
                let used_col = calc_col(input.A[d][used_idx], half);
                let not_used_col = calc_col(input.A[d][not_used_idx], half);
                let next = sum + used_col - not_used_col;
                let before_diff = (input.W as isize - sum as isize).abs();
                let after_diff = (input.W as isize - next as isize).abs();
                if col1 + col2 <= input.W * 2 || after_diff + 10 <= before_diff {
                    sum = next;
                    used[used_idx] = !used[used_idx];
                    used[not_used_idx] = !used[not_used_idx];
                    col1 += not_used_col - used_col;
                    col2 += used_col - not_used_col;
                    if sum <= input.W && sum > best_sum {
                        best_sum = sum;
                        best_used = used.clone();
                    }
                }
            }
            used_list.push(best_used);
        }

        let mut rows1 = vec![];
        let mut rows2 = vec![];
        let mut col_num_list1 = vec![];
        let mut col_num_list2 = vec![];
        let mut prefix_sum1 = vec![];
        let mut prefix_sum2 = vec![];
        for d in 0..input.D {
            let mut col_num1 = vec![];
            let mut col_num2 = vec![];
            let mut sum1 = 0;
            let mut sum2 = 0;
            let mut row1 = vec![];
            let mut row2 = vec![];
            for n in 0..input.N {
                let col_num = calc_col(input.A[d][n], half);
                if used_list[d][n] {
                    sum1 += col_num;
                    col_num1.push(col_num);
                    row1.push(n);
                } else {
                    sum2 += col_num;
                    col_num2.push(col_num);
                    row2.push(n);
                }
            }

            rows1.push(row1);
            rows2.push(row2);

            if sum1 < input.W {
                col_num1[0] += input.W - sum1;
            }
            if sum2 < input.W {
                col_num2[0] += input.W - sum2;
            }
            let mut n = 0;
            while sum1 > input.W {
                n += 1;
                n %= col_num1.len();
                if col_num1[n] == 1 {
                    continue;
                }
                col_num1[n] -= 1;
                sum1 -= 1;
            }
            while sum2 > input.W {
                n += 1;
                n %= col_num2.len();
                if col_num2[n] == 1 {
                    continue;
                }
                col_num2[n] -= 1;
                sum2 -= 1;
            }
            let mut prefix_col_num1 = vec![0; col_num1.len() + 1];
            let mut prefix_col_num2 = vec![0; col_num2.len() + 1];
            for i in 0..col_num1.len() {
                prefix_col_num1[i + 1] = prefix_col_num1[i] + col_num1[i];
            }
            for i in 0..col_num2.len() {
                prefix_col_num2[i + 1] = prefix_col_num2[i] + col_num2[i];
            }
            assert!(prefix_col_num1[col_num1.len()] == input.W);
            assert!(prefix_col_num2[col_num2.len()] == input.W);
            prefix_sum1.push(prefix_col_num1);
            prefix_sum2.push(prefix_col_num2);
            col_num_list1.push(col_num1);
            col_num_list2.push(col_num2);
        }
        State2 {
            rows1,
            rows2,
            col_num_list1,
            col_num_list2,
            prefix_sum1,
            prefix_sum2,
        }
    }
    fn output(&self, input: &Input, flag: bool) -> Vec<Vec<(usize, usize, usize, usize)>> {
        let mut ret = vec![];
        for d in 0..input.D {
            let mut out = vec![];
            for n in 0..input.N {
                if self.rows1[d].contains(&n) {
                    for (i, x) in self.rows1[d].iter().enumerate() {
                        if *x == n {
                            let row1 = 0;
                            let row2 = input.W / 2;
                            let col1 = self.prefix_sum1[d][i];
                            let col2 = self.prefix_sum1[d][i + 1];
                            if flag {
                                println!("{} {} {} {}", row1, col1, row2, col2);
                            }
                            out.push((row1, col1, row2, col2));
                            break;
                        }
                    }
                } else {
                    for (i, x) in self.rows2[d].iter().enumerate() {
                        if *x == n {
                            let row1 = input.W / 2;
                            let row2 = input.W;
                            let col1 = self.prefix_sum2[d][i];
                            let col2 = self.prefix_sum2[d][i + 1];
                            if flag {
                                println!("{} {} {} {}", row1, col1, row2, col2);
                            }
                            out.push((row1, col1, row2, col2));
                            break;
                        }
                    }
                }
            }
            ret.push(out);
        }
        ret
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
