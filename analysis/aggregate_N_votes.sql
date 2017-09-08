

CREATE TABLE asset_agg35votes_task1 
SELECT asset_id, name, 
	SUM(CASE WHEN answer_id = 1 THEN 1 ELSE 0 END) as smooth_count, 
	SUM(CASE WHEN answer_id = 2 THEN 1 ELSE 0 END) as feature_count,
	SUM(CASE WHEN answer_id = 3 THEN 1 ELSE 0 END) as star_count
	FROM 
		(SELECT asset_id, name, answer_id, created_at,
				@asset_rank := IF(@current_asset = asset_id, @asset_rank+1, 1) AS asset_rank, 
				@current_asset:=asset_id 
		FROM task1_full
		ORDER BY asset_id, created_at ASC
		) ranked 
WHERE asset_rank <= 35
GROUP BY asset_id;


/*
SELECT asset_id, created_at, answer_id,
              @asset_rank := IF(@current_asset = asset_id, @asset_rank + 1, 1) AS asset_rank,
              @current_asset := asset_id 
   FROM task1_expert_new
   ORDER BY asset_id, created_at ASC 
LIMIT 100;
/*

/*


SELECT asset_id, created_at, answer_id
FROM
 (SELECT asset_id, created_at, answer_id,
              @answer_rank := IF (@current_asset = asset_id, @answer_rank + 1, 1) AS answer_rank,
              @current_asset := asset_id 
   FROM task1_expert_new
   ORDER BY asset_id, created_at ASC
 ) ranked
WHERE answer_rank <= 2 limit 100;
*/

