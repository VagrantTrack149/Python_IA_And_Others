# Por anchura


grafo = {
        "A": ["D","C","B"], 
        "B": ["E","A"],
        "C": ["G","F","A"],
        "D": ["H","A"],
        "E": ["I","B"],
        "F": ["J","C"],
        "G": ["C"],
        "H": ["D"],
        "I": ["E"],
        "J": ["F"]
    } 


print("Muestra el grafo");

for key, lista in grafo.items():
    print(key)
    print(lista)
    
visitados = [];

pila = [];

origen = "A";
print("Lista de rcorrido en anchura");

pila.append(origen)

while pila:
    actual = pila.pop(0);
    if actual not in visitados:
        print("Valor actual", actual);
        visitados.append(actual);
    for lista in grafo[actual]:
        if lista not in visitados:
            pila.append(lista);

    

        
        
        
