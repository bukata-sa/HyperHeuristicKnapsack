################################################################################################
#               http://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html                   #
################################################################################################


def parse_mknap1(path_to_file):
    results = []
    with open(path_to_file) as file:
        number_of_cases = int(file.readline())
        file.readline()
        for case in range(number_of_cases):
            item_number, ksp_number, _ = file.readline().split(' ')
            item_number = int(item_number)
            ksp_number = int(ksp_number)
            costs = [float(cost) for cost in file.readline().split(' ')]
            weights = []
            for i in range(ksp_number):
                weights.append([int(weight) for weight in file.readline().split(' ')])
            sizes = [int(size) for size in file.readline().split(' ')]
            file.readline()
            results.append({
                'costs': costs,
                'weights': weights,
                'sizes': sizes
            })
    return results


def parse_mknap2(path_to_file):
    results = []
    with open(path_to_file) as file:
        while True:
            line = file.readline()
            if line == '':
                break
            if line[0] == '#':
                continue
            ksp_number, item_number = line.split(' ')
            ksp_number = int(ksp_number)
            item_number = int(item_number)
            costs = []
            while len(costs) < item_number:
                costs += [float(cost) for cost in file.readline().split(' ')]
            sizes = []
            while len(sizes) < ksp_number:
                sizes += [float(size) for size in file.readline().split(' ')]
            weights = []
            for i in range(ksp_number):
                current_weight = []
                while len(current_weight) < item_number:
                    current_weight += [int(weight) for weight in file.readline().split(' ')]
                weights.append(current_weight)
            file.readline()
            optimal = int(file.readline())
            file.readline()
            results.append({
                'costs': costs,
                'weights': weights,
                'sizes': sizes,
                'optimal': optimal
            })
    return results


def parse_mknapcb(path_to_file):
    results = []
    with open(path_to_file) as file:
        number_of_cases = int(file.readline())
        for case in range(number_of_cases):
            _, item_number, ksp_number, _, _ = file.readline().split(' ')
            item_number = int(item_number)
            ksp_number = int(ksp_number)

            costs = []
            while len(costs) < item_number:
                costs += [float(cost) for cost in file.readline().split(' ') if cost not in ('', '\n')]

            weights = []
            for i in range(ksp_number):
                current_weight = []
                while len(current_weight) < item_number:
                    current_weight += [int(weight) for weight in file.readline().split(' ') if weight not in ('', '\n')]
                weights.append(current_weight)

            sizes = []
            while len(sizes) < ksp_number:
                sizes += [float(size) for size in file.readline().split(' ') if size not in ('', '\n')]

            results.append({
                'costs': costs,
                'weights': weights,
                'sizes': sizes
            })
    return results


if __name__ == '__main__':
    # pprint(parse_mknap1("./resources/mknap1.txt"))
    # pprint(parse_mknap2("./resources/mknap2.txt"))
    parse_mknapcb("./resources/mknapcb1.txt")
