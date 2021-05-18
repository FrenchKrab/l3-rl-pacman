


def enter_values(valuedata_to_enter):
    """Asks the user to input values
    
    valuedata_to_enter must be a list of dictionnaries.
    These dictionnary must contain "name", "vartype", and can contain
    "desc" (optionnel) and "array" (if vartype=="arrayelement").
    
    Supported vartypes are:
    string, real, int, layers, arrayelement
    """
    result={}

    for i in range(len(valuedata_to_enter)):
        paramdata = valuedata_to_enter[i]
        print("--[{}/{}]---{} ({})---".format(i+1, len(valuedata_to_enter), paramdata["name"], paramdata["vartype"]))
        if "desc" in paramdata:
            print("Description: " + paramdata["desc"])
        final_value = None

        variable_type=paramdata["vartype"]
        #simple variable types
        if variable_type=="string" or variable_type=="real" or variable_type=="int":
            inputted_value=input("")
            if variable_type=="string":
                final_value = inputted_value
            elif inputted_value != "":
                if variable_type=="real":
                    final_value = float(inputted_value)
                if variable_type=="int":
                    final_value = int(inputted_value)
        #layer entry
        elif variable_type=="layers":
            stop_acquiring_layers=False
            final_value=[]
            print("Laissez vide pour arreter la saisie")
            while not stop_acquiring_layers:
                inputted_value = input("Couche cachée #{}, nombre de neurones: ".format(len(final_value)+1))
                if inputted_value != "":
                    final_value.append(int(inputted_value))
                else:
                    stop_acquiring_layers = True
        #select from array
        elif variable_type=="arrayelement":
            array = paramdata["array"]
            for i in range(len(array)):
                print("{}){}".format(i, array[i]))

            selected_element = -1
            while not (selected_element >= 0 and selected_element < len(array)):
                selected_element = int(input(""))
            final_value = array[selected_element]

        if final_value != None:
            result[paramdata["name"]] = final_value
    return result