## Dummy Summary Sheet


<p>
[]
```
```
</p>

<p>

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
    'DEVICE_ID':sqlalchemy.types.VARCHAR(50),
    'IP_ADDR':sqlalchemy.types.VARCHAR(100),
    'DOMAIN':sqlalchemy.types.VARCHAR(50),
    'PAGE_URL':sqlalchemy.types.VARCHAR(200),
    'REFERRER':sqlalchemy.types.VARCHAR(2000),
    'OS':sqlalchemy.types.VARCHAR(100),
    'OS_VER':sqlalchemy.types.VARCHAR(30),
    'BROWSER':sqlalchemy.types.VARCHAR(100),
    'BROWSER_VER':sqlalchemy.types.VARCHAR(30),
    'DEV':sqlalchemy.types.VARCHAR(100),
    'DEV_BRND':sqlalchemy.types.VARCHAR(100),
    'USER_AGENT':sqlalchemy.types.TEXT,
                },index=False)
```
</p>


<p>

[partion by example]

``` sql
# 파티션별 내림차순 순번  
SELECT ROW_NUMBER() OVER (PARTITION BY ~ ORDER BY ~ DESC) AS "NUM"
```
</p>
