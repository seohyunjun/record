## Dummy Summary Sheet
__NLL__
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

__NLL__
***

__FSS__

<p>
[intel pandas, sikit-learn 최적화 tool]

[intel OPEN API](https://www.intel.com/content/www/us/en/developer/tools/oneapi/ai-analytics-toolkit-download.html?operatingsystem=linux&distributions=docker)
</p>
<br>
<p>
[cuda10 cudnn7 python3.6]

[dockerHub](https://hub.docker.com/r/rogerchen/cuda10.0-cudnn7-py3.6)
</p>
<br>
<p>
[Deepo CUDA CUDNN Version Manage Using Docker]

- CUDA 11.3 까지 지원
- [GitHub](https://github.com/ufoym/deepo) 
</p>
<br>
<p>
[SKU]

- Stock Keeping Unit(재고 관리를 위한 최소 단위)
- 재고 추적을 위한 코드 (제품 색상, 크기 등 정보 ID) 

</p>
<br>
<p>
[Ubuntu GPU 확인]

[gpu monitoring tool](https://eungbean.github.io/2018/08/23/gpu-monitoring-tool-ubuntu/)

- nvidia-smi (gpu driver 확인)
- gpustat (pip install gpustat) -> 최애
- [gpumonitor](https://github.com/mountassir/gmonitor#building-from-source)
- glance (CPU, disk, network 모니터링,sudo apt-get install -y python-pip; sudo pip install glances[gpu]) 
</p>
<br>
<p>
[Tensorflow BERT Guide Document]

- [BERT로 텍스트 분류](https://www.tensorflow.org/text/tutorials/classify_text_with_bert)
- IMDB(영화 감상, 이진 분류 1: GOOD 0: BAD)
</p>
<br>
<p>
[Naver 영화 감상 BERT Fine-Tuning]

- [[Hands-on] Fine Tuning Naver Movie Review Sentiment Classification with KoBERT using GluonNLP](https://housekdk.gitbook.io/ml/ml/nlp/bert-fine-tuning-naver-movie)
</p>
<br>
<p>
[Text Labeling Tool]

- [라벨링 툴 목록](https://mangastorytelling.tistory.com/entry/Open-Source-데이터-라벨링-툴-목록-List-of-open-source-annotation-tools-for-ML)
- [doccanno](https://github.com/doccano/doccano) 
</p>
<br>
<p>
[CUDA OOM ERROR]

- [Pytorch 커뮤니티 답변](https://discuss.pytorch.kr/t/cuda-out-of-memory/216/6)
- OOM(Out of Memory)
- GPU RAM에 초과하는 Memory를 할당 할때 발생 (사용안하는 GPU 제거)

</p>

<br>
<p>
[selenium chrome 창 크기 조절]

- [selenium chrome 창 크기 조절](https://bskyvision.com/entry/python-2)

``` python
from selenium import webdriver
 
chromedriver = './chromedriver.exe'
driver = webdriver.Chrome(chromedriver)
driver.set_window_position(0, 0) # 위치 조절
driver.set_window_size(1000, 3000) # 윈도우 사이즈
```
</p>
<br>
<p>
[ARIMA Model 저장]

- [Persisting an ARIMA model](https://alkaline-ml.com/pmdarima/auto_examples/arima/example_persisting_a_model.html)
- pkl format으로 저장

```python
pickle_tgt = "arima.pkl"
try:
    # Pickle it
    joblib.dump(arima, pickle_tgt, compress=3)

    # Load the model up, create predictions
    arima_loaded = joblib.load(pickle_tgt)
    preds = arima_loaded.predict(n_periods=test.shape[0])
    print("Predictions: %r" % preds)

finally:
    # Remove the pickle file at the end of this example
    try:
        os.unlink(pickle_tgt)
    except OSError:
        pass
```
</p>
<br>
<p>
[docker commit]

- docker commit __<옵션>__ __<컨테이너 이름>__ __<이미지 이름>__:__<태그>__
</p>
<br>

<p>
[GPU Check in python]

``` python
def CheckGPUenv(gpu):
    if gpu=='tensorflow':
        print("Tensorflow Version : {tf.__version__}")
    
    if gpu=='pytorch':
        print(f"Pytorch Version  : {torch.__version__}")
        print(f"Pytorch Cudnn    : {torch.backends.cudnn.version()}")
        print(f"Pytorch Device   : {torch.cuda.get_device_name()}")
        print(f"Cuda Version     : {torch.version.cuda}")
        print(f"Cuda Available   : {torch.cuda.is_available()}") 
CheckGPUenv('pytorch')
```
</p>
<br>
<p>
[HTML Viewer in IOS]

- [shortcut](https://dev.to/setlock10/viewing-html-source-code-on-iphone-and-ipad-2879)

</p>
<br>
<p>
[RoBERTa + Zero-Shot]

- [Zero Shot Pipeline.ipynb](https://colab.research.google.com/drive/1jocViLorbwWIkTXKwxCOV9HLTaDDgCaw#scrollTo=uSoBpCpV6k4s)
- [zero-shot Topic Classification](https://jedleee.medium.com/zero-shot-topic-classification-fb92d1b33cfb)
- unseen class (such as a set of visual attributes or simply the class name) in order for a model to be able to predict that class without training data
    - training에 없는 데이터를 predict 하기 위한 방법
- Input 데이터에 부가적인 label 정보를 넣어 부가적인 정보를 통해 model이 데이터를 유추하게 만듦 
    - zebra ('tail', 'horse','white and black stripes')
    - large-scale training set에 labeling시 주로 사용
    - ex) image classification 시 이미지와 text 속성에 대한 encoding 진행 후 두 벡터를 __cosine_similarity__의 값으로 label을 잘 찾아가게 한다.
</p>
<br>
<p>
[Multi-label Classification]

- 다중 분류에 이어 label이 2차원 이상의 classification
- CS 문의 데이터를 예시)
    - 문의--서비스
    - 문의--기타문의 
    - 부가서비스--서비스 
- 1-depth 클래스에 대한 상호배타성을 유지 시킴 
- [Multi-Label Classification 정리](https://tonnykwon.github.io/blog/machine/2020-01-16-Multi-Label%20Classification/)
</p>
<br>
<p>
[pytorch 학습 모니터링]

- [training monitoring](https://gaussian37.github.io/dl-pytorch-observe/)
</p>
<br>
<p>
[python log]

- [python logging](https://kkangdda.tistory.com/m/64)

``` python
import logging
logger = logging.getLogger("log")
logger.setLevel(logging.INFO)
handler= logging.StreamHandler()

# set logging format
formatter = logging.Formatter('%(asctime)s|%(name)s|%(levelname)s:%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# set logging save  
streamHandler = logging.StreamHandler()
streamHandler.setLevel(logging.DEBUG)
streamHandler.setFormatter(formatter)

fileHandler = logging.FileHandler(loggfile_path)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(formatter)

logger.addHandler(streamHandler)
logger.addHandler(fileHandler)
```
</p>
<br>

<p>
[BERT Optimize Parameter BERT 원문 paper]

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
</p>
<br>
<p>
[Optimizer AdamW]

[AdamW](https://hiddenbeginner.github.io/deeplearning/paperreview/2019/12/29/paper_review_AdamW.html)
- loss update시 weight를 일정 비율 감소시킨다.
    - train 시 loss 발산을 방지   
    - weight decay: gradient descent update 시 이전 weight의 크기를 일정 비율 감소
    - 오버피팅 방지
</p>
<br>
<p>
[K-Fold vs Stratified-KFold]

[K-Fold | Stratified-KFold](https://guru.tistory.com/35)

- Stratified-KFold
    - 층화 표본 추출 
    - y값의 분포를 고려해 train set 구성
</p>
<br>
<p>
[데이터 불균형 (data imbalanced)]

[BERT data imbalanced embeding](https://fatemerhmi.github.io/files/Classification_of_imbalanced_dataset_using_BERT_embedding.pdf)
</p>
<br>
<p>
[서울대 NLP 연구실]

[knlp-snu](http://knlp.snu.ac.kr)
</p>
<br>
<p>
[Wordpiece Vocab 만들기]

- [Wordpiece Vocab ](https://monologg.kr/2020/04/27/wordpiece-vocab/)
- vocab quality 늘리기
</p>
<br>
<p>
[데이터 생성]

[kogpt 이용 데이터 생성](https://github.com/kakaobrain/kogpt)
</p>
<br>
<p>
[python 정확한 cost 측정 timeit]

[python timeit Document](https://docs.python.org/3/library/timeit.html)
</p>
<br>
<p>
[numba python->C compiler]

[numba](https://numba.pydata.org)
- JIT(Just In Time)
    - LLVM 컴파일러를 사용해 머신코드로 바꾸어 수치 연산 가속화
    - Cpython 사용 x 인터프리트 x
    - [예제](https://www.youtube.com/watch?time_continue=256&v=x58W9A2lnQc&feature=emb_logo) 
</p>
<br>
<p>
[Deploying Torchserve]

[Official TorchServe Deploy youTube](https://www.youtube.com/watch?v=jdE4hPf9juk&feature=youtu.be)
</p>
<br>
<p>
[Machine Learning Serving]

[Machine Learning Serving](https://github.com/bentoml/BentoML)
</p>
<br>
<p>
[SQL 보안]

[Autorize Manage](https://redapply.tistory.com/m/entry/SQL-접속시-접속정보-별도-보관해서-사용하기python)
</p>
<br>
<p>
[비동기 request]

[asyncio](https://www.youtube.com/watch?v=nFn4_nA_yk8)
</p>
<br>
<p>
[docker "" 이미지 삭제]

- docker rmi -f $(docker images -f “dangling=true” -q)
</p>
<br>
<p>
[docker 파일 복사]

- docker cp __container__:__remote/DIR/File__  ~ __client/DIR__
</p>
<br>
<p>
[GIT LAB CI/CD]

gitlab runner

```sh
[gitlab server]
docker run -d --name gitlab-runner --restart always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /srv/gitlab-runner/config:/etc/gitlab-runner:Z \
  gitlab/gitlab-runner:latest
```
</p>
<br>
<p>
[GITLAB CI/CD CD commit]

- ${} : gitlab variable 등록

```yml
build:
  image: python3.8/python
  stage: build
  script:
    - latexmk -pdf -pdflatex="xelatex -interaction=nonstopmode" -use-make *.tex
  artifacts:
    when: on_success
    paths:
      - ./*.pdf
    expire_in: 5 min # might not need this if deploy works


deploy:
  stage: deploy
  before_script:
    - 'which ssh-agent || ( apt-get update -qy && apt-get install openssh-client -qqy )'
    - eval `ssh-agent -s`
    - echo "${SSH_PRIVATE_KEY}" | tr -d '\r' | ssh-add - > /dev/null # add ssh ke
    - mkdir -p ~/.ssh
    - chmod 700 ~/.ssh
    - echo "$SSH_PUBLIC_KEY" >> ~/.ssh/id_rsa.pub
    - '[[ -f /.dockerenv ]] && echo -e "Host *\n\tStrictHostKeyChecking no\n\n" > ~/.ssh/config'
  script:
    - git config --global user.email "${CI_EMAIL}"
    - git config --global user.name "${CI_USERNAME}"
    - git add -f *.pdf # Force add PDF since we .gitignored it
    - git commit -m "Compiled PDF from $CI_COMMIT_SHORT_SHA [skip ci]" || echo "No changes, nothing to commit!"
    - git remote rm origin && git remote add origin git@gitlab.com:$CI_PROJECT_PATH.git
    - git push origin HEAD:$CI_COMMIT_REF_NAME # Pushes to the same branch as the trigger
```
</p>
<br>

<p>
[pre Commit]

[commit 전 사전 정의](https://pre-commit.com)
</p>
<br>
<p>
[GNN AI프렌즈]
[GNN](https://www.youtube.com/watch?v=rUmRlZzD_Uk&feature=youtu.be)
</p>
<br>
<p>
[2D->3D]
[shapeNet](https://shapenet.org)
</p>
<br>
<p>
[ViT Vision in Transformer]

[ViT Solve large scale dataset modeling](https://bnmy6581.tistory.com/87)
</p>
<br>
<p>
[Image Autolabeling paper]

[Auto labeling](https://arxiv.org/pdf/2007.07415.pdf)
</p>
<br>
<p>
[labelImg]

- Annotation Tool
- [labelImg](https://github.com/heartexlabs/labelImg)

</p>
<br>
<p>
[ViT Review in Korea Univ]

[Review](https://www.youtube.com/watch?v=NQIkPlCdSSI&feature=youtu.be)
</p>
<br>
<p>
[ViT Fine-Tuning Classification]

[hugging face](https://huggingface.co/blog/fine-tune-vit)
</p>
<br>
<p>
[ CLIP ]

- [CLIP](https://openai.com/blog/multimodal-neurons/): Connecting Text and Images
- Multi-Modal
- Text + Image 

</p>
<br>
<p>
[ReinforceLearning]

- [Free Lecture](https://simoninithomas.github.io/deep-rl-course/)

</p>
<br>
<p>
[NVIDIA Riva Speech]

[Nvidia RIVA](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/riva/containers/riva-speech)
</p>
<br>
<p>
[]
</p>
<br>
<br>
<p>
[]
</p>
<br>
__FSS__
***

