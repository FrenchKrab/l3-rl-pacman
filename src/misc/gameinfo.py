_ATARI_AVAILABLE_SETTINGS=[
    {"name":"frameskip", "vartype":"int"}
]

GAMES = [
    {
        "name":"tresor2d-v0",
        "displayed_name":"Recherche de trésor",
        "available_settings":[
            {"name":"width", "vartype":"int"},
            {"name":"height", "vartype":"int"},
            {"name":"generation", "desc":"Type de génération du labyrinthe", "vartype":"arrayelement", "array":["zigzag","random","empty"]}, 
        ]
    },
    {
        "name":"MsPacman-ram-v0",
        "displayed_name":"Pac-Man RAM",
        "available_settings":_ATARI_AVAILABLE_SETTINGS
    },
    {
        "name":"Breakout-ram-v0",
        "displayed_name":"Breakout RAM",
        "available_settings":_ATARI_AVAILABLE_SETTINGS
    },
    {
        "name":"CartPole-v0",
        "displayed_name":"CartPole",
        "available_settings":[]
    }
]



def get_game_list_numbered():
    """Returns a string containing all games numbered"""
    result = ""
    for i in range(len(GAMES)):
        result += "{}) {}\n".format(i, GAMES[i]["displayed_name"])
    return result[0:-1]     #removes the last line break character

def game_id_exists(id):
    """Tests if a games with id "id" exists"""
    return id >= 0 and id < len(GAMES)


def get_game_info(id):
    """Returns informations about the game with id "id" """
    return GAMES[id]