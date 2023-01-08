import sys

def generator_list(value):
    result = []
    for i in range(value):
        result.append(i)
    return result
result = generator_list(50)
print(sys.getsizeof(result))


def generator_list(value):
    result = []
    for i in range(value):
        yield i
result = generator_list(50)
print(sys.getsizeof(result))

result = list(generator_list(50))
print(sys.getsizeof(result))

test_list  =['치킨', '피자','삼겹살','족발','초밥','소주']
test_list[::2] 

"#You #Only #live #Once".replace('#',"")

0 ** 5

4 ** 5

name = "A"
price = 5600
number = 7
"{2} {1} {0}".format(number, name, price)