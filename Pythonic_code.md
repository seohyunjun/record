# Boostcamp AI Tech (feat.NAVER)

<sup>
2023 기초 복습 아는것도 다시
</sup>

## Pythonic Code
- 다양한 언어의 장점을 많이 가져왔다.
- 코드의 간.결.성.

### Why Pythonic Code?
- 남 코드에 대한 이해도
  - 많은 개발자들이 python 스타일로 코딩
- 효율
  - 단순 for loop append보다 list가 조금 더 빠르다
  - 익숙해지면 코드 간결성 증가
- 간지?
  - 쓰면 왠지 코드 잘 짜는 거처럼 보인다?!

## Contents
- split & join
  - split # parameter 기준으로 문자열 나누기
  - join  # parameter 기준으로 문자열 합치기  
- list comprehension
  - Nested list (중첩 list comprehesion)
  - Add Condition

    <sup>pprint : 횡 출력</sup>


- enumerate & zip
  - list의 element를 추출할때 index 추출
  - zip 병렬적으로 element 추출
- lambda & map & reduce
  - lambda 
    - Python 3부터는 권장하지 않으나 여전히 많이쓰임
    - PEP 8 
      - 테스트의 어려움  
      - 문서화 docstring 지원 미비
      - 코드해석의 어려움
      - 이름이 존재하지 않는 함수의 출현
  - map 
    - sequence 형태의 list에 값 input
    - list(map(function, list))
    - PEP 8 권장 X
  - reduce
    - map function과 달리 list에 똑같은 함수를 적용해서 통합
    - from functools import reduce
    - PEP 8 권장 X
    - 대용량의 데이터 다룰 때 종종 사용
    - reduct(lambda x, y: x+y, [1,2,3,4,5]) -> 15
- generator
  - Sequence형 자료형에서 데이터를 순서대로 출력하는 Object
  - address = iter(memory) memory의 주소를 순서대로 담는 그릇
  - next(address) 값 출력
  - element가 사용되는 시점에 값을 메모리에 반환
    - yield를 사용해 한번에 하나의 element만 반환
    - 메모리 주소 절약 가능 -> 대용량 데이터 핸들링시 사용
- asterisk
  - keyword argumets
  - 가변 길이 arguments (variable-length asterisk)
    - keyword의 개수를 모를 때 
      - def asterisk_test(a, b, *args) ~
        - _*args_ -> _tuple_
  - keyword variable-length (키워드의 명이 가변)
    - Parameter 이름을 따로 지정하지 않고 입력
    - _asterisk(*)_ 두개를 사용
    - 입력된 값은 _dict type_ 으로 사용가능
    - **kwargs -> _dict type_
  - __가변인자 순서__
    - (normal_parm1, named_parm1=1 , *args, **kwargs)
  - unpacking a container
    - asterisk_test(1, *(1,2,3))
      - *(1,2,3) -> (1,2,3)