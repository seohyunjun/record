## Dummy Summary Sheet

<br>
[pymysql mysql connect]

```python  
import pymysql
pymysql.install_as_MySQLdb()

import sqlalchemy as db
from sqlalchemy import create_engine

engine = db.create_engine('mysql+mysqldb://root:'+'passwd'+'@127.0.0.1:3306/db', encoding='utf-8')
conn = engine.connect()

# example
data.to_sql(name='data',con=conn,if_exists='append',dtype = {
    'DATE':sqlalchemy.types.CHAR(10),
    'EVENT':sqlalchemy.types.VARCHAR(5),
    'REG_DT':sqlalchemy.types.TIMESTAMP,
                },index=False)
```
</p>
<br>
<p>

[partion by example]

``` sql
# 파티션별 내림차순 순번  
SELECT ROW_NUMBER() OVER (PARTITION BY ~ ORDER BY ~ DESC) AS "NUM"
```

</p>
<br>
[Simpson Faces GAN]
<br>
[data(segmentation label, Mask image)]("https://www.kaggle.com/code/amirmohammadrostami/starter-simpsons-faces-e3007ec4-a")
<p>
[DB에 URL 저장시 Tip]
</p>
<br>
<p>

```sql
VARCHAR(512) CHARACTER SET 'ascii' COLLATE 'ascii_general_ci' NOT NULL
# Twitter URL 단축기 사용
``` 
</p>
<br>
<p>
[DATE FORMAT STR->DATE 날짜 변환 후 비교]

```sql
STR_TO_DATE(CONCAT('20220425','000000'), '%Y%m%d%H%i%s')
```
</p>
<br>
<p>
[DATE FORMAT STR->DATE 날짜 변환 후 비교]

```sql
STR_TO_DATE(CONCAT('20220425','000000'), '%Y%m%d%H%i%s')
```
</p>
<br>
<p>
[GROUP BY 별로 STR SUM, SEPARATOR:구분자]

```sql
CONCAT('\'', GROUP_CONCAT(DISTINCT VALUE SEPARATOR '\',\''), '\'') AS VALUE2
```
</p>
<br>
<p>
[Calculate lantitude, longtitude]

- 지구의 반지름을 6,400 km 로 가정
- 경도 1도의 거리 (km단위) = cos( 위도 ) * 6400 * 2 * 3.14 / 360
- 위도 1도의 거리는 아래와 같이 계산할 수 있고 대략 111 km
- 위도 1도의 거리 (km단위) = 6400 * 2 * 3.14 / 360

```python
X = ( cos( lat_1 ) * 6400 * 2 * 3.14 / 360 ) * | lat_1 - lng_2 |
Y = 111 * | lat_1 - lng_2 |
D = √ ( X² + Y² )
```
</p>
<br>
=============================NLL====================================

=============================FT====================================
<p>
[intel pandas, sikit-learn 최적화 tool]

[intel OPEN API](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html?operatingsystem=linux&distributions=docker)
</p>
<br>
<p>
[]

```sql
```
</p>