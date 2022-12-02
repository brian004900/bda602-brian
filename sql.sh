#!/bin/sh
sleep 10
DATABASE_USER=root
DATABASE_PWD=ma
DATABASE_NAME=baseball

DATABASE_TO_COPY_INTO="baseball"
DATABASE_FILE="baseball.sql"

mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "show databases;"

if mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb "use ${DATABASE_NAME}"
then
  echo "${DATABASE_NAME} exists"
else
  echo "${DATABASE_NAME} does not exist (creating it)"
  mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "CREATE DATABASE ${DATABASE_TO_COPY_INTO};"
  mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb ${DATABASE_TO_COPY_INTO} < ${DATABASE_FILE}
fi

mysql -u$DATABASE_USER -p$DATABASE_PWD -h mydb -e "
USE baseball;
create temporary table if not exists t1(
select battersInGame.game_id
, battersInGame.batter, local_date
, batter_counts.Hit as hit, batter_counts.atBat as atbat
from battersInGame
join game_temp
on battersInGame.game_id = game_temp.game_id
join batter_counts
on battersInGame.batter = batter_counts.batter and battersInGame.game_id = batter_counts.game_id
order by battersInGame.batter);

select *
from t1;

CREATE INDEX t1_idx_2 ON t1 (batter);
CREATE INDEX t1_idx_3 ON t1 (local_date);
CREATE INDEX t1_idx_4 on t1 (game_id);
CREATE INDEX t1_idx_5 ON t1 (game_id, batter);
CREATE INDEX t1_idx_6 ON t1 (local_date, batter);
CREATE UNIQUE INDEX t1_udx ON t1 (game_id, batter, local_date);

show tables;

create table if not exists rolling100(
select
      a.game_id
    , a.batter
    , a.local_date
    , case when SUM(b.atbat) = 0 then 0
    else cast(SUM(b.hit) as float )/cast(SUM(b.atbat) as float) end as rolling_batting
from t1 a
join t1 b
on b.batter = a.batter
where b.local_date between DATE_ADD(a.local_date, interval -100 day ) and DATE_ADD(a.local_date, interval -1 day)
group by a.game_id, a.batter, a.local_date
order by a.batter, a.local_date);"

Exit
