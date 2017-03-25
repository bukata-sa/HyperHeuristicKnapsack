import knapsack.hyper.genetic as hyper_gene

size = 6404180
weights = [382745, 799601, 909247, 729069, 467902, 44328, 34610, 698150, 823460, 903959, 853665, 551830, 610856, 670702,
           488960, 951111, 323046, 446298, 931161, 31385, 496951, 264724, 224916, 169684]
costs = [825594, 1677009, 1676628, 1523970, 943972, 97426, 69666, 1296457, 1679693, 1902996, 1844992, 1049289, 1252836,
         1319836, 953277, 2067538, 675367, 853655, 1826027, 65731, 901489, 577243, 466257, 369261]
start = [0] * len(weights)
optimal = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1]

result = 0
for i in range(0, 50):
    optimal_funcs = hyper_gene.minimize(10, weights=weights, costs=costs, size=size, included=start)
    current = hyper_gene.fitness_hyper_ksp(optimal_funcs, weights=weights, costs=costs, size=size, included=start)
    print(13549094 - current)
    result += 13549094 - current
    print(result)
print(result)
