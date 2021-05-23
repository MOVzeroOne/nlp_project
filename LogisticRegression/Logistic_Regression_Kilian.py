import json

if __name__ == '__main__':
    jsonfile = open('/Users/kiliankramer/Desktop/All_Beauty.json', 'r')
    json_string = jsonfile.read()
    json_string = "[" + json_string.replace("}\n{", "},{") + "]"
    reviews = json.loads(json_string)
    dictionary = {}
    dictionary2 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    dictionary3 = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for i in range(len(reviews)):
        if 'reviewText' in reviews[i]:
            text = reviews[i]['reviewText'];
            text = text.replace(".", "")
            text = text.replace(",", "")
            text = text.replace(";", "")
            text = text.replace("!", "")
            text = text.replace("?", "")
            words = text.split()
            for j in words:
                if j in dictionary:
                    dictionary[j] += 1
                else:
                    dictionary[j] = 1
            if 'overall' in reviews[i]:
                dictionary3[int(reviews[i]['overall'])] += 1;
                if 'is' in text:
                    dictionary2[int(reviews[i]['overall'])] += 1;

    # Trying to find some feature words:

    print(dictionary['is'])
    print(dictionary['good'])
    print(dictionary['well'])
    print(dictionary['great'])
    print(dictionary['awesome'])
    print(dictionary['fantastic'])
    print(dictionary['nice'])
    print(dictionary['beautiful'])

    print(dictionary['sick'])
    print(dictionary['better'])
    print(dictionary['much'])
    print(dictionary['very'])

    print(dictionary['bad'])
    print(dictionary['don\'t'])
    print(dictionary['doesn\'t'])
    print(dictionary['not'])
    print(dictionary['shit'])
    print(dictionary['ugly'])
    print(dictionary['waste'])
    print(dictionary['burns'])
    print("hi")
