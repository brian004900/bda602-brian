create table historical_avg(
select
      game_id
    , batter
,CASE WHEN atBat = 0 THEN 0
    else cast(SUM(Hit) as float )/cast(SUM(atBat) as float) end as batting_average
from batter_counts
group by batter
order by batter);


create table annual_batting_avg(
select battersInGame.game_id, battersInGame.batter, EXTRACT(year FROM game.local_date) as year
, CASE WHEN batter_counts.atBat = 0 THEN 0
    else cast(SUM(batter_counts.Hit) as float )/cast(SUM(batter_counts.atBat) as float) end as batting_average
from battersInGame
join game
on battersInGame.game_id = game.game_id
join batter_counts
on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
group by year, battersInGame.batter
order by battersInGame.batter, year);


CREATE TEMPORARY TABLE t1 (
select battersInGame.game_id
, battersInGame.batter, local_date
, batter_counts.Hit as hits, batter_counts.atBat as atbats
from battersInGame
join game_temp
on battersInGame.game_id = game_temp.game_id
join batter_counts
on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
order by battersInGame.batter);

CREATE UNIQUE INDEX t1_idx
ON t1 (game_id, batter, local_date, hits, atbats);


create table rolling_avg(
select
      a.game_id
    , a.batter
    , a.local_date
    , CASE WHEN SUM(b.atBats) = 0 THEN 0
    else cast(SUM(b.hits) as float )/cast(SUM(b.atBats) as float) end as rolling_batting
from t1 a
join t1 b
on b.batter = a.batter
where b.local_date between DATE_ADD(a.local_date, INTERVAL -99 DAY) and a.local_date
group by a.game_id, a.local_date, a.batter,  a.hits, a.atbats
order by batter, a.local_date);