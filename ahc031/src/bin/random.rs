#![allow(non_snake_case)]
#![allow(unused_imports)]
#![allow(unused_macros)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::nonminimal_bool)]
#![allow(clippy::neg_multiply)]
#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]
#![allow(dead_code)]
#![allow(clippy::len_zero)]

use itertools::Itertools;
use std::collections::BTreeSet;
use std::ops::RangeBounds;

use proconio::input;
use rand::prelude::*;
use rand_pcg::Pcg64Mcg;
use rustc_hash::FxHashSet;

const INF: usize = 1_usize << 60;

fn main() {
    get_time();
    let input = read_input();
    let mut rng = Pcg64Mcg::new(12345);

    let mut best_bin_width = vec![];
    let mut best_assignments = vec![vec![vec![]]];
    let mut best_score = INF;

    // 仕切りの数を0から最大N-1まで順番に確認
    'a: for sep in 0..input.N {
        let elapsed = get_time();
        if elapsed > 1.2 {
            break;
        }
        // 各仕切り毎の探索時間の上限
        while get_time() - elapsed < 0.3 {
            let bin_width = make_partition(&input, sep, &mut rng);
            // ビンパッキング問題を解く
            // 入れられるビンの中で、高さが最も低いビンに入れる貪欲
            let (ok, assignments) = solve_bin_packing(&bin_width, &input);
            let (ok, assignments) = if ok {
                (ok, assignments)
            } else {
                solve_bin_packing2(&bin_width, &input)
            };
            let score = calc_score(&input, &bin_width, &assignments);

            if score < best_score {
                best_score = score;
                best_bin_width = bin_width;
                best_assignments = assignments;
            }
            if sep == 0 || ok {
                // 仕切り無しのときは1通りの詰め方しかないので、強制的に仕切りを増やす
                // 解が見つかったら、打ち切って、仕切りを増やす
                continue 'a;
            }
        }
    }

    // 最大の仕切り数で時間いっぱい探索
    let mut sep = best_bin_width.len() - 1;
    eprintln!("Sep: {}", sep);
    sep = sep.max(1); // 仕切り無しは探索不要
    while get_time() < 2.9 {
        let bin_width = make_partition(&input, sep, &mut rng);
        let (_, assignments) = solve_bin_packing(&bin_width, &input);
        let score = calc_score(&input, &bin_width, &assignments);
        if score < best_score {
            best_score = score;
            best_bin_width = bin_width.clone();
            best_assignments = assignments;
        }
        let (_, assignments) = solve_bin_packing2(&bin_width, &input);
        let score = calc_score(&input, &bin_width, &assignments);
        if score < best_score {
            best_score = score;
            best_bin_width = bin_width;
            best_assignments = assignments;
        }
    }

    output(&input, &best_bin_width, &best_assignments);
    eprintln!("Score: {}", best_score);
    eprintln!("Elapsed: {}", (get_time() * 1000.0) as usize);

    // ローカルテスター移植
    #[cfg(feature = "local")]
    {
        let out_file = std::env::args().nth(1).unwrap();
        let output = std::fs::read_to_string(&out_file).unwrap_or_else(|_| {
            eprintln!("no such file: {}", out_file);
            std::process::exit(1)
        });
        let out = parse_output(&input, &output);
        let (score, err) = match out {
            Ok(out) => compute_score(&input, &out),
            Err(err) => (0, err),
        };
        eprintln!("Tester: {}", score);
        eprintln!("Error: {}", err);
    }
}

fn make_partition(input: &Input, sep: usize, rng: &mut Pcg64Mcg) -> Vec<usize> {
    // 仕切りの位置をランダムに決める
    let mut positions = vec![0, input.W];
    let mut set = FxHashSet::default();
    while set.len() < sep {
        let pos = rng.gen_range(1..input.W);
        if !set.contains(&pos) {
            positions.push(pos);
            set.insert(pos);
        }
    }
    positions.sort();
    // 各ビンの幅を計算
    let mut bin_width = vec![];
    for i in 1..sep + 2 {
        let width = positions[i] - positions[i - 1];
        bin_width.push(width);
    }
    bin_width
}

fn solve_bin_packing(bin_width: &[usize], input: &Input) -> (bool, Vec<Vec<Vec<usize>>>) {
    let mut bin = vec![vec![0; bin_width.len()]; input.D];
    let mut assignment = vec![vec![vec![]; bin_width.len()]; input.D];
    let mut ok = true;
    for d in 0..input.D {
        // 大きい方から順番に入れていく
        for (n, &a) in input.a[d].iter().enumerate().rev() {
            let mut height = INF;
            let mut id = 0;
            for i in 0..bin_width.len() {
                let h = ceil(a, bin_width[i]) + bin[d][i];
                // 最も低いビンを指定
                if h < height {
                    height = h;
                    id = i;
                }
            }
            bin[d][id] = height;
            // ビンに入らない場合はNG
            if height > input.W {
                ok = false;
            }
            // 各ビンに何を割り当てるか記録
            assignment[d][id].push(n);
        }
        // 空ビンがあればNG
        for i in 0..bin_width.len() {
            if bin[d][i] == 0 {
                ok = false;
            }
        }
    }
    (ok, assignment)
}

fn solve_bin_packing2(bin_width: &[usize], input: &Input) -> (bool, Vec<Vec<Vec<usize>>>) {
    let mut bin = vec![vec![0; bin_width.len()]; input.D];
    let mut assignment = vec![vec![vec![]; bin_width.len()]; input.D];
    let mut ok = true;
    for d in 0..input.D {
        // 大きい方から順番に入れていく
        for (n, &a) in input.a[d].iter().enumerate().rev() {
            let mut ratio = 1e18;
            let mut height = 0;
            let mut id = 0;
            for i in 0..bin_width.len() {
                let h = ceil(a, bin_width[i]) + bin[d][i];
                // 縦横比が小さいビンを指定
                let r = (h as f64 / bin_width[i] as f64).min(bin_width[i] as f64 / h as f64);
                if h < input.W && r < ratio {
                    ratio = r;
                    height = h;
                    id = i;
                }
            }
            bin[d][id] = height;
            // ビンに入らない場合はNG
            if height > input.W {
                ok = false;
            }
            // 各ビンに何を割り当てるか記録
            assignment[d][id].push(n);
        }
        // 空ビンがあればNG
        for i in 0..bin_width.len() {
            if bin[d][i] == 0 {
                ok = false;
            }
        }
    }
    (ok, assignment)
}

fn calc_score(input: &Input, bin_width: &[usize], assignments: &[Vec<Vec<usize>>]) -> usize {
    let mut bin = vec![FxHashSet::default(); bin_width.len()];
    let mut score = 1;

    for d in 0..input.D {
        let mut next_bin = vec![FxHashSet::default(); bin_width.len()];
        for (i, &width) in bin_width.iter().enumerate() {
            let mut height = 0;
            for (j, &n) in assignments[d][i].iter().rev().enumerate() {
                let mut h = ceil(input.a[d][n], width);
                if height + h > input.W {
                    // ビンに詰め切れない場合はペナルティ
                    h = input.W - height;
                    score += (input.a[d][n] - h * width) * 100;
                } else if j == assignments[d][i].len() - 1 {
                    // あるビンの最後に詰めるときはパーティション不要
                    h = input.W - height;
                } else {
                    next_bin[i].insert(height + h);
                }
                height += h;
            }
        }
        // パーティションのコスト計算
        if d > 0 {
            // 縦パーティションのコスト計算
            // 隣接した2つのビンが空の日があって、前後の日のいずれかで、空でなければ、縦のパーティションがコストになる
            for i in 1..bin_width.len() {
                if assignments[d - 1][i - 1].is_empty()
                    && assignments[d - 1][i].is_empty()
                    && (!assignments[d][i - 1].is_empty() || !assignments[d][i].is_empty())
                {
                    score += input.W;
                }
                if assignments[d][i - 1].is_empty()
                    && assignments[d][i].is_empty()
                    && (!assignments[d - 1][i - 1].is_empty() || !assignments[d - 1][i].is_empty())
                {
                    score += input.W;
                }
            }

            // 横パーティションのコスト計算
            for i in 0..bin_width.len() {
                // 前日と当日の横の仕切りを全て足す
                bin[i].remove(&input.W);
                next_bin[i].remove(&input.W);
                score += bin[i].len() * bin_width[i];
                score += next_bin[i].len() * bin_width[i];
                // 前日と当日で等しい仕切りは引く
                for h in bin[i].iter() {
                    if next_bin[i].contains(h) {
                        score -= 2 * bin_width[i];
                    }
                }
            }
        }
        std::mem::swap(&mut bin, &mut next_bin);
    }
    score
}

fn output(input: &Input, bin_width: &[usize], assignments: &[Vec<Vec<usize>>]) {
    // ビン幅の累積和を求める（ビンの左右の位置を事前計算）
    let mut prefix_sum_bin_width = vec![0; bin_width.len() + 1];
    for i in 0..bin_width.len() {
        prefix_sum_bin_width[i + 1] = prefix_sum_bin_width[i] + bin_width[i];
    }

    for d in 0..input.D {
        let mut ans = vec![(0, 0, 0, 0); input.N];
        for (i, &width) in bin_width.iter().enumerate() {
            let mut height = 0;
            for (j, &n) in assignments[d][i].iter().rev().enumerate() {
                let mut h = ceil(input.a[d][n], width);
                // 最後にビンに入れるものは、無駄な仕切りを追加しないようにする
                if j == assignments[d][i].len() - 1 {
                    h = input.W - height;
                }
                ans[n] = (
                    height,
                    prefix_sum_bin_width[i],
                    height + h,
                    prefix_sum_bin_width[i + 1],
                );
                height += h;
            }
        }
        for n in 0..input.N {
            println!("{} {} {} {}", ans[n].0, ans[n].1, ans[n].2, ans[n].3);
        }
    }
}

fn ceil(a: usize, b: usize) -> usize {
    (a + b - 1) / b
}

fn read_input() -> Input {
    input! {
        W: usize,
        D: usize,
        N: usize,
        a: [[usize; N]; D]
    }

    Input { W, D, N, a }
}

pub fn get_time() -> f64 {
    static mut STIME: f64 = -1.0;
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let ms = t.as_secs() as f64 + t.subsec_nanos() as f64 * 1e-9;
    unsafe {
        if STIME < 0.0 {
            STIME = ms;
        }
        // ローカル環境とジャッジ環境の実行速度差はget_timeで吸収しておくと便利
        #[cfg(feature = "local")]
        {
            (ms - STIME) * 0.55
        }
        #[cfg(not(feature = "local"))]
        {
            ms - STIME
        }
    }
}

pub trait SetMinMax {
    fn setmin(&mut self, v: Self) -> bool;
    fn setmax(&mut self, v: Self) -> bool;
}
impl<T> SetMinMax for T
where
    T: PartialOrd,
{
    fn setmin(&mut self, v: T) -> bool {
        *self > v && {
            *self = v;
            true
        }
    }
    fn setmax(&mut self, v: T) -> bool {
        *self < v && {
            *self = v;
            true
        }
    }
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
    pub W: usize,
    pub D: usize,
    pub N: usize,
    pub a: Vec<Vec<usize>>,
}

impl std::fmt::Display for Input {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{} {} {}", self.W, self.D, self.N)?;
        for i in 0..self.D {
            writeln!(f, "{}", self.a[i].iter().join(" "))?;
        }
        Ok(())
    }
}

pub fn parse_input(f: &str) -> Input {
    let f = proconio::source::once::OnceSource::from(f);
    input! {
        from f,
        W: usize, D: usize, N: usize,
        a: [[usize; N]; D],
    }
    Input { W, D, N, a }
}

pub fn read<T: Copy + PartialOrd + std::fmt::Display + std::str::FromStr, R: RangeBounds<T>>(
    token: Option<&str>,
    range: R,
) -> Result<T, String> {
    if let Some(v) = token {
        if let Ok(v) = v.parse::<T>() {
            if !range.contains(&v) {
                Err(format!("Out of range: {}", v))
            } else {
                Ok(v)
            }
        } else {
            Err(format!("Parse error: {}", v))
        }
    } else {
        Err("Unexpected EOF".to_owned())
    }
}

pub struct Output {
    pub out: Vec<Vec<(usize, usize, usize, usize)>>,
}

pub fn parse_output(input: &Input, f: &str) -> Result<Output, String> {
    let mut out = mat![(0, 0, 0, 0); input.D; input.N];
    let mut ss = f.split_whitespace();
    for d in 0..input.D {
        for i in 0..input.N {
            let i0 = read(ss.next(), 0..=input.W)?;
            let j0 = read(ss.next(), 0..=input.W)?;
            let i1 = read(ss.next(), 0..=input.W)?;
            let j1 = read(ss.next(), 0..=input.W)?;
            if i0 >= i1 || j0 >= j1 {
                return Err(format!(
                    "Invalid rectangle: {} {} {} {} (d = {}, k = {})",
                    i0, j0, i1, j1, d, i
                ));
            }
            out[d][i] = (i0, j0, i1, j1);
        }
    }
    if ss.next().is_some() {
        return Err("Too many output".to_owned());
    }
    Ok(Output { out })
}

pub fn gen(seed: u64, fix_D: Option<usize>, fix_N: Option<usize>, fix_E: Option<usize>) -> Input {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed ^ 677);
    let W = 1000;
    let mut D = rng.gen_range(5i32..=50) as usize;
    if let Some(fix_D) = fix_D {
        D = fix_D;
    }
    let mut N = rng.gen_range(5i32..=50) as usize;
    if let Some(fix_N) = fix_N {
        N = fix_N;
    }
    let e = rng.gen_range(500..=5000) as f64 / 10000.0;
    let mut E = ((W * W) as f64 * e * e).round() as usize;
    if let Some(fix_E) = fix_E {
        E = fix_E;
    }
    let mut a = vec![vec![]; D];
    for d in 0..D {
        let total = rng.gen_range((W * W - E * 3 / 2) as i32..=(W * W - E / 2) as i32);
        let mut set = std::collections::BTreeSet::new();
        set.insert(0);
        set.insert(total);
        while set.len() <= N {
            set.insert(rng.gen_range(1..total));
        }
        let set = set.into_iter().collect::<Vec<_>>();
        for i in 0..N {
            a[d].push((set[i + 1] - set[i]) as usize);
        }
        a[d].sort();
    }
    Input { W, D, N, a }
}

pub fn compute_score(input: &Input, out: &Output) -> (i64, String) {
    let (mut score, err, _) = compute_score_details(input, &out.out);
    if err.len() > 0 {
        score = 0;
    }
    (score, err)
}

pub fn compute_score_details(
    input: &Input,
    out: &[Vec<(usize, usize, usize, usize)>],
) -> (i64, String, Vec<(usize, usize, usize, usize, bool)>) {
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
                    return (
                        0,
                        format!("Rectangles {} and {} overlap on day {}.", q, p, d),
                        vec![],
                    );
                }
            }
        }
        let mut hs2 = BTreeSet::new();
        let mut vs2 = BTreeSet::new();
        for k in 0..input.N {
            let (i0, j0, i1, j1) = out[d][k];
            let b = (i1 - i0) * (j1 - j0);
            if input.a[d][k] > b {
                score += 100 * (input.a[d][k] - b) as i64;
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
                        if change.len() > 0
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
                        if change.len() > 0
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
                        if change.len() > 0
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
                        if change.len() > 0
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
    (score + 1, String::new(), change)
}
