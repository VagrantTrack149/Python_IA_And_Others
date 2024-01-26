# En profundidad


grafo = {
        "A": ["D","C","B"], 
        "B": ["E"],
        "C": ["G","F"],
        "D": ["H"],
        "E": ["I"],
        "F": ["J"]
    } 


print("Muestra el grafo");

for key, lista in grafo.items():
    print(key)
    print(lista)
    
visitados = [];

pila = [];

origen = "A";
print("Lista de rcorrido en profundidad");

pila.append(origen)

while pila:
    actual = pila.pop();
    if actual not in visitados:
        print("Valor actual", actual);
        visitados.append(actual);
    if actual not in grafo:
        continue;
    for lista in grafo[actual]:
        if lista not in visitados:
            pila.append(lista);

print(visitados)