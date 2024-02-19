#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]

use std::{
    cmp::Reverse,
    collections::{BTreeMap, BTreeSet, BinaryHeap, VecDeque},
};

use itertools::Itertools;
use proconio::{
    input,
    marker::{Chars, Usize1},
};
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha20Rng;
use superslice::Ext;

fn main() {
    let mut stdin =
        proconio::source::line::LineSource::new(std::io::BufReader::new(std::io::stdin()));
    macro_rules! input(($($tt:tt)*) => (proconio::input!(from &mut stdin, $($tt)*)));

    input! {
        N: usize,
        M: usize,
        eps: f64,
    }

    let mut minos = vec![];
    for _ in 0..M {
        input! {
            d: usize,
            coords: [(usize, usize); d]
        }
        let mut coords2 = vec![];
        let mut height = 0;
        let mut width = 0;
        for coord in coords.iter() {
            let row = coord.0;
            let col = coord.1;
            height.chmax(row + 1);
            width.chmax(col + 1);
            coords2.push(Coord::new(row, col));
        }
        let mino = Mino {
            coords: coords2,
            height,
            width,
        };
        minos.push(mino);
    }

    // Local
    // input! {
    //     _ps: [(usize, usize); M],
    //     ans: [[i32; N]; N],
    //     es: [f64; 2*N*N]
    // }
    // let ans = ans.into_iter().flatten().collect_vec();
    // let ans = DynamicMap2d::new(ans, N);
    // Local

    let mut turn = 0;
    let mut cost = 0.0_f64;

    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(0);

    let mut mino_num = 0;
    for mino in minos.iter() {
        mino_num += mino.coords.len();
    }

    let mut cnt = 0;
    let mut mp = DynamicMap2d::new(vec![None; N * N], N);

    'outer: while cnt < mino_num {
        let mut i = rng.gen_range(0..N);
        let mut j = rng.gen_range(0..N);
        loop {
            if mp[Coord::new(i, j)].is_none() {
                break;
            }
            i = rng.gen_range(0..N);
            j = rng.gen_range(0..N);
        }
        let now = Coord::new(i, j);

        // let out = query(&[now], &ans, eps, &es, &mut turn, &mut cost); // Local
        make_query(&[now]);
        input! {out:usize}

        if out == 0 {
            mp[now] = Some(0);
            continue;
        }
        cnt += out;
        if cnt >= mino_num {
            break;
        }
        mp[now] = Some(out);
        let mut Q = VecDeque::new();
        Q.push_back(now);

        while let Some(u) = Q.pop_front() {
            for adj in ADJ {
                let v = u + adj;
                if !v.in_map(N, N) {
                    continue;
                }
                if mp[v].is_some() {
                    continue;
                }
                // let out = query(&[v], &ans, eps, &es, &mut turn, &mut cost);  // Local
                make_query(&[v]);
                input! {out:usize}

                if out == 0 {
                    mp[v] = Some(0);
                } else {
                    mp[v] = Some(out);
                    Q.push_back(v);
                    cnt += out;
                    if cnt >= mino_num {
                        break 'outer;
                    }
                }
            }
        }
    }

    make_answer(&mp);
    input! {out:usize}

    let score = (1e6 * cost.max(1.0 / N as f64)).round() as usize;
    eprintln!("Cost: {cost}");
    eprintln!("Score: {score}");
}

fn calc_prob(mut out: f64, k: usize, eps: f64) -> Vec<f64> {
    let base = 3;
    let num = (base - 1) * k;
    let mut probs = vec![0.0; num + 1];
    let std = (k as f64 * eps * (1.0 - eps)).sqrt();
    let delta = std * 0.1;
    if out == 0.0 {
        out -= std;
    }
    for i in 0..=num {
        let mean = (k as f64 - i as f64) * eps + (i as f64) * (1.0 - eps);
        let diff = (mean - out).abs();
        probs[i] = e(diff + delta, std) - e(diff - delta, std);
    }

    let sum = probs.iter().sum::<f64>().min(1.0);
    if sum == 0.0 {
        probs = vec![0.0; num + 1];
        probs[num] = 1.0;
    } else {
        let sum_inv = 1.0 / sum;
        for i in 0..=num {
            probs[i] *= sum_inv;
        }
    }

    let pattern_num = base.pow(k as u32);
    let mut cnt_pattern = vec![vec![]; num + 1];
    for i in 0..pattern_num {
        let mut n = i;
        let mut s = 0;
        while n > 0 {
            s += n % base;
            n /= base;
        }
        cnt_pattern[s].push(i);
    }
    let mut prob = vec![0.0; base];
    for i in 0..=num {
        for p in cnt_pattern[i].iter() {
            prob[p % base] += probs[i] / cnt_pattern[i].len() as f64;
        }
    }
    prob
}

fn e(x: f64, std: f64) -> f64 {
    0.5 + 0.5 * erf(x / std / 2.0_f64.sqrt())
}

fn erf(x: f64) -> f64 {
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    const P: f64 = 0.3275911;

    let t = 1.0 / (1.0 + P * x);
    1.0 - (((((A5 * t + A4) * t) + A3) * t + A2) * t + A1) * t * (-x * x).exp()
}

#[derive(Debug)]
struct Mino {
    coords: Vec<Coord>,
    height: usize,
    width: usize,
}

fn query(
    coords: &[Coord],
    ans: &DynamicMap2d<i32>,
    eps: f64,
    es: &[f64],
    turn: &mut usize,
    cost: &mut f64,
) -> usize {
    let k = coords.len() as f64;
    *turn += 1;
    *cost += 1.0 / k.sqrt();
    if k == 1.0 {
        return ans[coords[0]] as usize;
    }
    let mut vs = 0.0;
    for coord in coords.iter() {
        vs += ans[*coord] as f64;
    }
    let mean = (k - vs) * eps + vs * (1.0 - eps);
    let std = (k * eps * (1.0 - eps)).sqrt();
    let ret = mean + std * es[*turn];
    let ret = ret.round() as usize;
    ret.max(0)
}

fn make_query(coords: &[Coord]) {
    let mut v = vec![];
    v.push(coords.len());
    for coord in coords.iter() {
        v.push(coord.row);
        v.push(coord.col);
    }
    let mut query = "q ".to_string();
    query += v.iter().join(" ").as_str();
    println!("{query}");
}

fn make_answer(mp: &DynamicMap2d<Option<usize>>) {
    let N = mp.size;
    let mut coords = vec![];
    for i in 0..N {
        for j in 0..N {
            let coord = Coord::new(i, j);
            if let Some(x) = mp[coord] {
                if x > 0 {
                    coords.push(coord);
                }
            }
        }
    }
    let mut v = vec![];
    v.push(coords.len());
    for coord in coords.iter() {
        v.push(coord.row);
        v.push(coord.col);
    }
    let mut ans = "a ".to_string();
    ans += v.iter().join(" ").as_str();
    println!("{ans}");
}

#[derive(Debug, Clone, Copy)]
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

pub const ADJACENT4: [CoordDiff; 4] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
];

pub const ADJACENT6: [CoordDiff; 6] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
];

pub const ADJACENT8: [CoordDiff; 8] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
    CoordDiff::new(3, 0),
    CoordDiff::new(3, 1),
];

pub const ADJACENT10: [CoordDiff; 10] = [
    CoordDiff::new(0, 0),
    CoordDiff::new(0, 1),
    CoordDiff::new(1, 0),
    CoordDiff::new(1, 1),
    CoordDiff::new(2, 0),
    CoordDiff::new(2, 1),
    CoordDiff::new(3, 0),
    CoordDiff::new(3, 1),
    CoordDiff::new(4, 0),
    CoordDiff::new(4, 1),
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
