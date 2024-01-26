grafo = {
    "A5":["B5","A6"],
    "A6":["A5","A7"],
    "A7":["B7"],
    "A9":["B9"],
    "B1":["C1"],
    "B3":["C3"],
    "B5":["C5","A5"],
    "B7":["A7","C7"],
    "B9":["A9","C9"],
    "C1":["B1","D1"],
    "C2":["C1","C3"],
    "C3":["B3","D3"],
    "C5":["B5"],
    "C7":["B7","D7"],
    "C9":["B9","D9"],
    "D1":["C1"],
    "D3":["C3","E3"],
    "D7":["C7","E7"],
    "D9":["E9","C9"],
    "E3":["D3","F3"],
    "E5":["F5","E6"],
    "E6":["E5","E7"],
    "E7":["E6","E8","D7"],
    "E8":["E9"],
    "E9":["E8","D9","F9"],
    "F1":["G1","F2"],
    "F2":["F1","F3"],
    "F3":["E3","G3","F2","F4"],
    "F4":["F3","F5"],
    "F5":["F4","E5","G5"],
    "F9":["E9","G9"],
    "G1":["H1","F1"],
    "G3":["F3","H3"],
    "G5":["F5","H5"],
    "G7":["G8"],
    "G8":["G7","G9"],
    "G9":["F9","G8"],
    "H1":["G1"],
    "H3":["G3","I3"],
    "H5":["I5","G5"],
    "I3":["H3","J3"],
    "I5":["H5","J5"],
    "I6":["I5","I7"],
    "I7":["I6","J7","I8"],
    "I8":["I7","I9"],
    "I9":["I8","J9"],
    "J1":["J2"],
    "J2":["J3","J1"],
    "J3":["I3"],
    "J5":["I5"],
    "J7":["I7"],
    "J9":["I9"]
}

print("Muestra el grafo")

for key, lista in grafo.items():
    print(key)
    print(lista)

visitados = []

pila = []

origen = "H1"  
destino = "E9" 

print("Lista de recorrido en profundidad iterativo")

pila.append(origen)

while pila:
    actual = pila.pop()
    if actual not in visitados:
        print("Valor actual", actual)
        visitados.append(actual)

    if actual == destino: 
        break

    if actual not in grafo:
        continue

    # En lugar de la recursión, empujamos los vecinos no visitados a la pila
    for vecino in grafo[actual]:
        if vecino not in visitados:
            pila.append(vecino)

print(visitados)