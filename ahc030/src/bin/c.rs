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

use itertools::{concat, Itertools};
use proconio::{
    input,
    marker::{Chars, Usize1},
};
use rand::{prelude::*, SeedableRng};
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, Normal};
use superslice::Ext;
use svg::node::element::path::Data;
use svg::node::element::{Circle, Group, Line, Path, Rectangle, Style, Text, Title};
use svg::node::Text as TextContent;
use svg::Document;

fn main() {
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

    input! {
        ps: [(usize, usize); M],
        ans: [[i32; N]; N],
        es: [f64; 2*N*N]
    }

    let mut turn = 0;
    let mut cost = 0.0;
    let ans = ans.into_iter().flatten().collect_vec();
    let ans = DynamicMap2d::new(ans, N);

    let init = vec![vec![0.5, 0.43, 0.05, 0.01, 0.01]; N * N];
    let mut distribution = DynamicMap2d::new(init, N);

    for i in (0..N).step_by(1) {
        for j in (0..N).step_by(1) {
            let mut coords = vec![];
            let now = Coord::new(i, j);
            for adj in ADJACENT4 {
                let mut next = now + adj;
                if !next.in_map(N, N) {
                    let ni = next.row % N;
                    let nj = next.col % N;
                    next = Coord::new(ni, nj);
                }
                coords.push(next);
            }

            let out = query(&coords, &ans, eps, &es, &mut turn, &mut cost);
            let prob = calc_prob(out, coords.len(), eps);
            for coord in coords.iter() {
                let mut p = vec![];
                for i in 0..prob.len() {
                    p.push(prob[i] * distribution[*coord][i]);
                }
                let sum_inv = 1.0 / p.iter().sum::<f64>().min(1.0);
                for i in 0..prob.len() {
                    p[i] *= sum_inv;
                }
                distribution[*coord] = p;
            }
        }
    }

    eprintln!("Cost: {}", cost / (N as f64 * N as f64));

    let mut prob = DynamicMap2d::new(vec![0.0; N * N], N);
    let mut mx = 0.0_f64;
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            mx = mx.max(1.0 - distribution[now][0]);
        }
    }
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            prob[now] = (1.0 - distribution[now][0]) / mx * 100.0;
        }
    }

    let mut mp = DynamicMap2d::new(vec![None; N * N], N);
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            if prob[now] < 0.005 {
                mp[now] = Some(false)
            } else {
                if prob[now] > 50.0 {
                    mp[now] = Some(true);
                    continue;
                }
                let out = query(&[now], &ans, eps, &es, &mut turn, &mut cost);
                if out > 0.0 {
                    mp[now] = Some(true);
                } else {
                    mp[now] = Some(false)
                }
            }
        }
    }

    let mut ac = true;
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            if ans[now] > 0 && mp[now] == Some(false) {
                ac = false;
            }
            if ans[now] == 0 && mp[now] == Some(true) {
                ac = false;
            }
        }
    }
    let score = (1e6 * cost.max(1.0 / N as f64)).round() as usize;
    eprintln!("Cost: {}", cost / (N as f64 * N as f64));
    eprintln!("Score: {score}");
    if ac {
        eprintln!("AC");
    } else {
        eprintln!("WA");
    }
    vis(&distribution, &ans, N);
    vis2(&mp, &ans, N);
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
) -> f64 {
    let k = coords.len() as f64;
    *turn += 1;
    *cost += 1.0 / k.sqrt();
    if k == 1.0 {
        return ans[coords[0]] as f64;
    }
    let mut vs = 0.0;
    for coord in coords.iter() {
        vs += ans[*coord] as f64;
    }
    let mean = (k - vs) * eps + vs * (1.0 - eps);
    let std = (k * eps * (1.0 - eps)).sqrt();
    let ret = mean + std * es[*turn];
    ret.max(0.0)
}

fn vis(distribution: &DynamicMap2d<Vec<f64>>, ans: &DynamicMap2d<i32>, N: usize) {
    let size = 600.0;
    let mut doc = doc(size, size);
    doc = doc.add(Style::new(format!(
        "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
        8
    )));
    let d = size / N as f32;
    let mut mx = 0.0_f64;
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            let p = 1.0 - distribution[now][0];
            mx = mx.max(p);
        }
    }
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);
            let p = (1.0 - distribution[now][0]) / mx;
            let mut rec = rect(j as f32 * d, i as f32 * d, d, d, "green").set("fill-opacity", p);
            if ans[now] > 0 {
                rec = rec.set("stroke", "red");
            }
            doc = doc.add(rec);
            doc = doc.add(txt(
                j as f32 * d + d / 2.0,
                i as f32 * d + d / 2.0,
                format!("{:.2}", p * 100.0).as_str(),
            ));
        }
    }

    let vis = format!("<html><body>{}</body></html>", doc);
    std::fs::write("vis.html", vis).unwrap();
}

fn vis2(mp: &DynamicMap2d<Option<bool>>, ans: &DynamicMap2d<i32>, N: usize) {
    let size = 600.0;
    let mut doc = doc(size, size);
    doc = doc.add(Style::new(format!(
        "text {{text-anchor: middle; dominant-baseline: central; font-size: {}}}",
        8
    )));
    let d = size / N as f32;
    let mut cnt = 0;
    for i in 0..N {
        for j in 0..N {
            let now = Coord::new(i, j);

            let color = match mp[now] {
                Some(true) => "black",
                Some(false) => "white",
                None => {
                    cnt += 1;
                    "lightgray"
                }
            };

            let mut rec = rect(j as f32 * d, i as f32 * d, d, d, color);
            if ans[now] > 0 {
                rec = rec.set("stroke", "red");
            }
            doc = doc.add(rec);
        }
    }
    // println!("gray cnt: {cnt}/{}", N * N);
    let vis = format!("<html><body>{}</body></html>", doc);
    std::fs::write("vis2.html", vis).unwrap();
}

#[macro_export]
macro_rules! mat {
	($($e:expr),*) => { Vec::from(vec![$($e),*]) };
	($($e:expr,)*) => { Vec::from(vec![$($e),*]) };
	($e:expr; $d:expr) => { Vec::from(vec![$e; $d]) };
	($e:expr; $d:expr $(; $ds:expr)+) => { Vec::from(vec![mat![$e $(; $ds)*]; $d]) };
}

#[derive(Clone, Debug)]
pub struct Input {
    pub n: usize,
    pub m: usize,
    pub eps: f64,
    pub ts: Vec<Vec<(usize, usize)>>,
    pub ps: Vec<(usize, usize)>,
    pub ans: Vec<Vec<i32>>,
    pub es: Vec<f64>,
}

const DIJ: [(usize, usize); 4] = [(0, 1), (1, 0), (0, !0), (!0, 0)];

fn gen(
    seed: u64,
    fix_N: Option<usize>,
    fix_M: Option<usize>,
    fix_eps: Option<f64>,
    rng: &mut rand_chacha::ChaCha20Rng,
) -> Input {
    // let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
    let mut n = rng.gen_range(10i32..=20) as usize;
    if let Some(v) = fix_N {
        n = v;
    }
    let mut m = rng.gen_range(2i32..=(n * n / 20) as i32) as usize;
    if let Some(v) = fix_M {
        m = v;
    }
    let mut eps = rng.gen_range(1..=20) as f64 / 100.0;
    if let Some(v) = fix_eps {
        eps = v;
    }
    let avg = (rng.gen_range((n * n / 5) as i32..=(n * n / 2) as i32) as usize / m).max(4);
    let delta = rng.gen_range(0..=avg as i32 - 4) as usize;
    let mut ts = vec![];
    for _ in 0..m {
        let size = rng.gen_range((avg - delta) as i32..=(avg + delta) as i32) as usize;
        let mut used = mat![false; n; n];
        let mut list = vec![];
        list.push((n / 2, n / 2));
        used[n / 2][n / 2] = true;
        let mut adj = vec![];
        for (di, dj) in DIJ {
            let i2 = n / 2 + di;
            let j2 = n / 2 + dj;
            adj.push((i2, j2));
            used[i2][j2] = true;
        }
        while list.len() < size {
            let p = rng.gen_range(0..adj.len() as i32) as usize;
            let (i, j) = adj.remove(p);
            list.push((i, j));
            for (di, dj) in DIJ {
                let i2 = i + di;
                let j2 = j + dj;
                if i2 < n && j2 < n && !used[i2][j2] {
                    adj.push((i2, j2));
                    used[i2][j2] = true;
                }
            }
        }
        let min_i = list.iter().map(|x| x.0).min().unwrap();
        let min_j = list.iter().map(|x| x.1).min().unwrap();
        for x in &mut list {
            x.0 -= min_i;
            x.1 -= min_j;
        }
        list.sort();
        ts.push(list);
    }
    let mut ans = mat![0; n; n];
    let mut ps = vec![];
    for p in 0..m {
        let max_i = ts[p].iter().map(|x| x.0).max().unwrap();
        let max_j = ts[p].iter().map(|x| x.1).max().unwrap();
        let di = rng.gen_range(0..(n - max_i) as i32) as usize;
        let dj = rng.gen_range(0..(n - max_j) as i32) as usize;
        for &(i, j) in &ts[p] {
            ans[i + di][j + dj] += 1;
        }
        ps.push((di, dj));
    }
    let mut es = vec![];
    for _ in 0..n * n * 2 {
        es.push(rng.sample(rand_distr::StandardNormal));
    }
    Input {
        n,
        m,
        eps,
        ts,
        ps,
        ans,
        es,
    }
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

const MARGIN: f32 = 10.0;

pub fn doc(height: f32, width: f32) -> Document {
    Document::new()
        .set(
            "viewBox",
            (
                -MARGIN,
                -MARGIN,
                width + 2.0 * MARGIN,
                height + 2.0 * MARGIN,
            ),
        )
        .set("width", width + MARGIN)
        .set("height", height + MARGIN)
        .set("style", "background-color:#F2F3F5")
}

pub fn rect(x: f32, y: f32, w: f32, h: f32, fill: &str) -> Rectangle {
    Rectangle::new()
        .set("x", x)
        .set("y", y)
        .set("width", w)
        .set("height", h)
        .set("fill", fill)
}

pub fn txt(x: f32, y: f32, text: &str) -> Text {
    Text::new()
        .add(TextContent::new(text))
        .set("x", x)
        .set("y", y)
        .set("fill", "black")
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
