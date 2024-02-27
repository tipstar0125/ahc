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
    collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque},
};

fn main() {
    let start = std::time::Instant::now();
    #[cfg(feature = "local")]
    {
        let seed = 1;
        eprintln!("Seed: {seed}");
        rnd::init(seed);
    }

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

const TARGET: isize = 5e17 as isize;
const TURN_MAX: usize = 50;

fn solve() {
    input! {
        N: usize,
        AB: [(isize, isize); N]
    }

    let mut best_score = calc_score(AB[0]);
    let mut operations = vec![];
    let mut turn = 0;
    let mut now = AB;
    while turn < TURN_MAX - 15 {
        let mut cands = vec![];
        let mut set = HashSet::new();
        while set.len() < 40 {
            let i = rnd::gen_range(0, N);
            let j = rnd::gen_range(0, N);
            if i == j {
                continue;
            }
            if set.contains(&(i, j)) {
                continue;
            }
            set.insert((i, j));
            let mut next = now.clone();
            let score = calc_score(next[0]);
            let ret = op(next[i], next[j]);
            next[i] = ret;
            next[j] = ret;
            let next_score = play_out(next, turn + 1);
            if score < next_score {
                cands.push((next_score, i, j));
            }
        }

        if cands.is_empty() {
            break;
        }
        cands.sort();
        let (score, i, j) = cands[cands.len() - 1];
        if score >= best_score {
            best_score = score;
            operations.push((i, j));
            let ret = op(now[i], now[j]);
            now[i] = ret;
            now[j] = ret;
        } else {
            break;
        }
        turn += 1;
    }

    best_score = calc_score(now[0]);

    while turn < TURN_MAX - 1 {
        let mut cands = vec![];
        for i in 1..N {
            for j in i + 1..N {
                if i == j {
                    continue;
                }
                let ret = op(now[i], now[j]);
                let ret2 = op(now[0], ret);
                let score = calc_score(ret2);
                if score > best_score {
                    cands.push((score, i, j));
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort();
        let (score, i, j) = cands[cands.len() - 1];
        if score > best_score {
            best_score = score;
            let ret = op(now[i], now[j]);
            let ret2 = op(now[0], ret);
            operations.push((i, j));
            operations.push((0, i));
            now[0] = ret2;
            now[i] = ret2;
            now[j] = ret;
        }
        turn += 2;
    }
    output(&operations);
    eprintln!("score: {}", best_score);
    eprintln!("turn: {}", turn);
}

fn play_out(mut AB: Vec<(isize, isize)>, mut turn: usize) -> usize {
    let N = AB.len();
    let mut best_score = calc_score(AB[0]);
    while turn < TURN_MAX {
        let mut cands = vec![];
        for i in 1..N {
            for j in 1..N {
                if i == j {
                    continue;
                }
                let ret = op(AB[i], AB[j]);
                let ret2 = op(AB[0], ret);
                let score = calc_score(ret2);
                if score > best_score {
                    cands.push((score, i, j));
                }
            }
        }
        if cands.is_empty() {
            break;
        }
        cands.sort();
        let (score, i, j) = cands[cands.len() - 1];
        if score > best_score {
            best_score = score;
            let ret = op(AB[i], AB[j]);
            let ret2 = op(AB[0], ret);
            AB[0] = ret2;
            AB[i] = ret2;
            AB[j] = ret;
        }
        turn += 1;
    }
    best_score
}

fn op(u: (isize, isize), v: (isize, isize)) -> (isize, isize) {
    ((u.0 + v.0) / 2, (u.1 + v.1) / 2)
}

fn calc_score(ab: (isize, isize)) -> usize {
    let v1 = (ab.0 - TARGET).abs();
    let v2 = (ab.1 - TARGET).abs();
    let mx = max!(v1, v2) as f64;
    let score = 2e6 - 1e5 * (mx + 1.0).log10();
    score.floor() as usize
}

fn output(operations: &[(usize, usize)]) {
    println!("{}", operations.len());
    for &(a, b) in operations.iter() {
        println!("{} {}", a + 1, b + 1);
    }
}

#[macro_export]
macro_rules! max {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::max($x, max!($( $y ),+))
    }
}
#[macro_export]
macro_rules! min {
    ($x: expr) => ($x);
    ($x: expr, $( $y: expr ),+) => {
        std::cmp::min($x, min!($( $y ),+))
    }
}

mod rnd {
    static mut S: usize = 0;
    static MAX: usize = 1e9 as usize;

    #[inline]
    pub fn init(seed: usize) {
        unsafe {
            if seed == 0 {
                let t = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs() as usize;
                S = t
            } else {
                S = seed;
            }
        }
    }
    #[inline]
    pub fn gen() -> usize {
        unsafe {
            if S == 0 {
                init(0);
            }
            S ^= S << 7;
            S ^= S >> 9;
            S
        }
    }
    #[inline]
    pub fn gen_range(a: usize, b: usize) -> usize {
        gen() % (b - a) + a
    }
    #[inline]
    pub fn gen_bool() -> bool {
        gen() & 1 == 1
    }
    #[inline]
    pub fn gen_range_isize(a: usize) -> isize {
        let mut x = (gen() % a) as isize;
        if gen_bool() {
            x *= -1;
        }
        x
    }
    #[inline]
    pub fn gen_range_neg_wrapping(a: usize) -> usize {
        let mut x = gen() % a;
        if gen_bool() {
            x = x.wrapping_neg();
        }
        x
    }
    #[inline]
    pub fn gen_float() -> f64 {
        ((gen() % MAX) as f64) / MAX as f64
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

fn join_to_string<T: std::string::ToString>(v: &[T], sep: &str) -> String {
    v.iter()
        .fold("".to_string(), |s, x| s + x.to_string().as_str() + sep)
}

#[macro_export]
macro_rules! input {
    () => {};
    (mut $var:ident: $t:tt, $($rest:tt)*) => {
        let mut $var = __input_inner!($t);
        input!($($rest)*)
    };
    ($var:ident: $t:tt, $($rest:tt)*) => {
        let $var = __input_inner!($t);
        input!($($rest)*)
    };
    (mut $var:ident: $t:tt) => {
        let mut $var = __input_inner!($t);
    };
    ($var:ident: $t:tt) => {
        let $var = __input_inner!($t);
    };
}

#[macro_export]
macro_rules! __input_inner {
    (($($t:tt),*)) => {
        ($(__input_inner!($t)),*)
    };
    ([$t:tt; $n:expr]) => {
        (0..$n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    };
    ([$t:tt]) => {{
        let n = __input_inner!(usize);
        (0..n).map(|_| __input_inner!($t)).collect::<Vec<_>>()
    }};
    (chars) => {
        __input_inner!(String).chars().collect::<Vec<_>>()
    };
    (bytes) => {
        __input_inner!(String).into_bytes()
    };
    (usize1) => {
        __input_inner!(usize) - 1
    };
    ($t:ty) => {
        $crate::read::<$t>()
    };
}

#[macro_export]
macro_rules! println {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w).unwrap()
        })
    };
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::writeln!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {
        $crate::write(|w| {
            use std::io::Write;
            std::write!(w, $($arg)*).unwrap()
        })
    };
}

#[macro_export]
macro_rules! flush {
    () => {
        $crate::write(|w| {
            use std::io::Write;
            w.flush().unwrap()
        })
    };
}

pub fn read<T>() -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Debug,
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDIN: RefCell<StdinLock<'static>> = RefCell::new(stdin().lock());
    }

    STDIN.with(|r| {
        let mut r = r.borrow_mut();
        let mut s = vec![];
        loop {
            let buf = r.fill_buf().unwrap();
            if buf.is_empty() {
                break;
            }
            if let Some(i) = buf.iter().position(u8::is_ascii_whitespace) {
                s.extend_from_slice(&buf[..i]);
                r.consume(i + 1);
                if !s.is_empty() {
                    break;
                }
            } else {
                s.extend_from_slice(buf);
                let n = buf.len();
                r.consume(n);
            }
        }
        std::str::from_utf8(&s).unwrap().parse().unwrap()
    })
}

pub fn write<F>(f: F)
where
    F: FnOnce(&mut std::io::BufWriter<std::io::StdoutLock>),
{
    use std::cell::RefCell;
    use std::io::*;

    thread_local! {
        pub static STDOUT: RefCell<BufWriter<StdoutLock<'static>>> =
            RefCell::new(BufWriter::new(stdout().lock()));
    }

    STDOUT.with(|w| f(&mut w.borrow_mut()))
}
