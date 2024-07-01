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
    collections::{vec_deque, BTreeMap, BinaryHeap, HashMap, VecDeque},
    fmt::format,
    fs,
};

use itertools::Itertools;
use proconio::{input, marker::Chars};
use rand::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use serde::{Deserialize, Serialize};
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::io::Write;

fn solve(filename: &str) {
    let input_filename = "in/".to_string() + filename;
    let output_filename = "out/".to_string() + filename;
    let input_fn = fs::read_to_string(input_filename).unwrap();
    let input: Input = serde_json::from_str(&input_fn).unwrap();
    eprintln!(
        "{} Turn: {}, Monster: {}",
        filename,
        input.num_turns,
        input.monsters.len()
    );
    let init_state = State::new(&input);
    let mut best_score = 0;
    let mut best_ans = vec![];
    let mut t = 0;
    while t < input.num_turns {
        let mut state = init_state.clone();
        while !state.is_done(&input) {
            if let Some((target_id, actions, coord)) = state.get_next_target(&input, t) {
                state.advance(target_id, actions, coord, &input);
            } else {
                break;
            }
        }
        if state.gold > best_score {
            best_score = state.gold;
            best_ans = state.ans;
        }
        t += 10;
    }
    eprintln!("Score: {}", best_score);
    output(&output_filename, &best_ans);
}

fn main() {
    let args: Vec<_> = env::args().collect();
    if args.len() > 1 {
        let filename = &args[1];
        solve(filename);
    } else {
        for i in 1..=25 {
            let filename = format!("{i:0>3}.json");
            solve(&filename);
        }
    }
}

fn calc_para(level: usize, base: usize, coeff: usize) -> usize {
    ((base as f64) * (1.0 + (level as f64) * (coeff as f64) / 100.0)) as usize
}

fn calc_diff2(x0: usize, y0: usize, x1: usize, y1: usize) -> usize {
    let diff_x = x0 as isize - x1 as isize;
    let diff_y = y0 as isize - y1 as isize;
    let diff_x2 = (diff_x * diff_x) as usize;
    let diff_y2 = (diff_y * diff_y) as usize;
    diff_x2 + diff_y2
}

fn ceil(x: usize, y: usize) -> usize {
    (x + y - 1) / y
}

#[derive(Debug, Clone)]
struct State {
    level: usize,
    exp: usize,
    gold: usize,
    speed: usize,
    speed2: usize,
    power: usize,
    range2: usize,
    x: usize,
    y: usize,
    monsters_status: Vec<bool>,
    ans: Vec<(usize, usize, usize)>,
}

impl State {
    fn new(input: &Input) -> Self {
        let speed = calc_para(0, input.hero.base_speed, input.hero.level_speed_coeff);
        let power = calc_para(0, input.hero.base_power, input.hero.level_power_coeff);
        let range = calc_para(0, input.hero.base_range, input.hero.level_range_coeff);

        State {
            level: 0,
            exp: 0,
            gold: 0,
            speed,
            speed2: speed * speed,
            power,
            range2: range * range,
            x: input.start_x,
            y: input.start_y,
            monsters_status: vec![false; input.monsters.len()],
            ans: vec![],
        }
    }
    fn advance(
        &mut self,
        target_id: usize,
        actions: Vec<(usize, usize, usize)>,
        coord: (usize, usize),
        input: &Input,
    ) {
        self.ans.extend(actions);
        self.x = coord.0;
        self.y = coord.1;
        self.monsters_status[target_id] = true;
        self.exp += input.monsters[target_id].exp;
        self.gold += input.monsters[target_id].gold;
        while self.level_up(input) {}
    }
    fn get_next_target(
        &self,
        input: &Input,
        t: usize,
    ) -> Option<(usize, Vec<(usize, usize, usize)>, (usize, usize))> {
        let mut infos = vec![];
        let mut actions = vec![vec![]; input.monsters.len()];
        for id in 0..input.monsters.len() {
            if self.monsters_status[id] {
                continue;
            }
            let mut x = self.x;
            let mut y = self.y;
            let mx = input.monsters[id].x;
            let my = input.monsters[id].y;
            let mut diff2 = calc_diff2(x, y, mx, my);
            let mut turn = 0;
            while diff2 > self.range2 {
                turn += 1;
                if diff2 <= self.speed2 {
                    x = mx;
                    y = my;
                    actions[id].push((0, x, y));
                    break;
                }
                let ratio = (diff2 as f64 / self.speed2 as f64).sqrt();
                let dx = (((mx as isize - x as isize) as f64) / ratio) as isize;
                let dy = (((my as isize - y as isize) as f64) / ratio) as isize;
                x = (x as isize + dx).clamp(0, input.width as isize) as usize;
                y = (y as isize + dy).clamp(0, input.height as isize) as usize;
                actions[id].push((0, x, y));
                diff2 = calc_diff2(x, y, mx, my);
            }
            for _ in 0..ceil(input.monsters[id].hp, self.power) {
                turn += 1;
            }
            if self.ans.len() + turn >= input.num_turns {
                continue;
            }
            let score = if self.ans.len() <= t {
                input.monsters[id].exp
            } else {
                input.monsters[id].gold
            };
            infos.push((score as f64 / turn as f64, id));
        }
        if infos.is_empty() {
            return None;
        }
        infos.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let target_id = infos[0].1;
        let mut action = actions[target_id].clone();
        let mut last_coord = (self.x, self.y);
        if !action.is_empty() {
            let (_, lx, ly) = action[action.len() - 1];
            last_coord = (lx, ly);
        }

        for _ in 0..ceil(input.monsters[target_id].hp, self.power) {
            action.push((1, target_id, 0));
        }
        Some((target_id, action, last_coord))
    }
    fn is_level_up(&self, exp: usize) -> bool {
        let necessary_exp = 1000 + self.level * (self.level + 1) * 50;
        exp >= necessary_exp
    }
    fn level_up(&mut self, input: &Input) -> bool {
        let necessary_exp = 1000 + self.level * (self.level + 1) * 50;
        if self.exp < necessary_exp {
            return false;
        }
        self.level += 1;
        self.exp -= necessary_exp;
        self.speed = calc_para(
            self.level,
            input.hero.base_speed,
            input.hero.level_speed_coeff,
        );
        self.power = calc_para(
            self.level,
            input.hero.base_power,
            input.hero.level_power_coeff,
        );
        let range = calc_para(
            self.level,
            input.hero.base_range,
            input.hero.level_range_coeff,
        );
        self.speed2 = self.speed * self.speed;
        self.range2 = range * range;
        true
    }
    fn is_done(&self, input: &Input) -> bool {
        self.ans.len() >= input.num_turns
    }
}

fn output(filename: &str, ans: &[(usize, usize, usize)]) {
    let mut actions = String::new();
    for &(a, b, c) in ans.iter() {
        if a == 0 {
            actions += format!(
                "{{\"type\":\"{}\",\"target_x\":{},\"target_y\":{}}},",
                "move", b, c
            )
            .as_str();
        } else {
            actions += format!("{{\"type\":\"{}\",\"target_id\":{}}},", "attack", b).as_str();
        }
    }
    actions.pop();
    let out = format!("{{\"moves\":[{}]}}", actions);
    // println!("{}", out);
    let mut file = File::create(filename).unwrap();
    file.write_all(out.as_bytes()).unwrap();
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Input {
    num_turns: usize,
    width: usize,
    height: usize,
    start_x: usize,
    start_y: usize,
    hero: Hero,
    monsters: Vec<Monster>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Hero {
    base_speed: usize,
    base_power: usize,
    base_range: usize,
    level_speed_coeff: usize,
    level_power_coeff: usize,
    level_range_coeff: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Monster {
    x: usize,
    y: usize,
    hp: usize,
    exp: usize,
    gold: usize,
}

#[derive(Debug, Clone)]
pub struct TimeKeeper {
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
    pub fn get_time(&self) -> f64 {
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
