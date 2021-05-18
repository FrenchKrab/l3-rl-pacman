import gym
import gym_tresor2d
from agents.dqn_agent_v2 import DQNAgent
from agents.qlearning_agent import QLearningAgent
import numpy as np
import glob
import os

import misc.gameinfo as gameinfo
import misc.console_helper as console_helper
import misc.pacman_tools as pacman_tools


#--------CONSTANTES---------


ACTIVATION_FUNCTIONS=['relu','tanh','sigmoid','hard_sigmoid','exponential','linear','elu','softmax','selu','softplus','softsign']
BOOL_CHOICE_ARRAY = [False, True]


SAVE_FOLDER="saves/"

ENVS_DQN_POSTPROCESS_TRAINPARAMS={
    "MsPacman-ram-v0":{
        "init_postprocess_data_fun" : pacman_tools.mspacman_init_data,
        "reset_postprocess_fun" : pacman_tools.ms_pacman_reset_post_process,
        "step_postprocess_fun" : pacman_tools.ms_pacman_step_post_process,
        "state_preprocess_fun" : pacman_tools.normalize_ram
    },
    "Breakout-ram-v0":{
        "state_preprocess_fun" : pacman_tools.normalize_ram
    }
}


QLEARNING_HYPERPARAMS=[
    {"name":"learning_rate", "vartype":"real"}, 
    {"name":"discount_factor", "vartype":"real"}
]
DQN_HYPERPARAMS=[
    {"name":"learning_rate", "vartype":"real"}, 
    {"name":"discount_factor", "vartype":"real"},
    {"name":"memory_size", "vartype":"int", "desc":"Taille de l'experience replay"},
    {"name":"hidden_layers", "vartype":"layers"},
    {"name":"activation_function", "vartype":"arrayelement", "array":ACTIVATION_FUNCTIONS, "desc":"Fonction d'activation commune à tous les layers"}
]

DESC_EPISODE_COUNT="Nombre d'épisodes sur lequel lancer l'agent"
DESC_EPSILON="Valeur initale d'epsilon"
DESC_VERBOSE="0=silencieux, 1=infos en fin d'épisode, 2=info en temps réel (conseillé sauf en cas d'affichage textuel du jeu), 3=affichage de l'état du jeu (debug)"
DESC_SAVE_FILE="Laissez vide pour ne pas sauvegarder"
DESC_VISUALIZE="Voir graphiquement l'état du jeu"
DESC_TARGET_FPS="Vitesse d'affichage désirée du jeu en images par seconde, laissez vide ou à -1 pour la vitesse maximale"

QLEARNING_TRAINPARAMS=[
    {"name":"episode_count", "vartype":"int", "desc":DESC_EPISODE_COUNT}, 
    {"name":"epsilon", "vartype":"real", "desc":DESC_EPSILON},
    {"name":"epsilon_min", "vartype":"real"},
    {"name":"epsilon_decay", "vartype":"real"},
    {"name":"decay_epsilon_on_episodes", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":"Si vrai, le decay d'epsilon ne se produira qu'en fin d'épisode"},
    {"name":"save_file", "vartype":"string", "desc":DESC_SAVE_FILE},
    {"name":"save_interval", "vartype":"int"},
    {"name":"visualize", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":DESC_VISUALIZE},
    {"name":"verbose", "vartype":"int", "desc":DESC_VERBOSE}
]
DQN_TRAINPARAMS=[
    {"name":"episode_count", "vartype":"int", "desc":DESC_EPISODE_COUNT}, 
    {"name":"batch_size", "vartype":"int", "desc":"Taille du minibatch"},
    {"name":"replay_size_required", "vartype":"int"},
    {"name":"epsilon", "vartype":"real", "desc":DESC_EPSILON},
    {"name":"epsilon_min", "vartype":"real"},
    {"name":"epsilon_decay", "vartype":"real"},
    {"name":"decay_epsilon_on_episodes", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":"Si vrai, le decay d'epsilon ne se produira qu'en fin d'épisode"},
    {"name":"save_file", "vartype":"string", "desc":DESC_SAVE_FILE},
    {"name":"save_interval", "vartype":"int"},
    {"name":"visualize", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":DESC_VISUALIZE},
    {"name":"verbose", "vartype":"int", "desc":DESC_VERBOSE}
]


QLEARNING_TESTPARAMS=[
    {"name":"episode_count", "vartype":"int", "desc":DESC_EPISODE_COUNT}, 
    {"name":"epsilon", "vartype":"real", "desc":DESC_EPSILON},
    {"name":"visualize", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":DESC_VISUALIZE},
    {"name":"target_fps", "vartype":"int", "desc":DESC_TARGET_FPS},
    {"name":"verbose", "vartype":"int", "desc":DESC_VERBOSE}
]
DQN_TESTPARAMS=[
    {"name":"episode_count", "vartype":"int", "desc":DESC_EPISODE_COUNT}, 
    {"name":"epsilon", "vartype":"real", "desc":DESC_EPSILON},
    {"name":"visualize", "vartype":"arrayelement", "array":BOOL_CHOICE_ARRAY, "desc":DESC_VISUALIZE},
    {"name":"target_fps", "vartype":"int", "desc":DESC_TARGET_FPS},
    {"name":"verbose", "vartype":"int", "desc":DESC_VERBOSE}
]


#--------Choix du jeu---------


is_game_chosen = False
chosent_game_id = 0

while not is_game_chosen:
    print("-----------\nSur quel environnement voulez-vous travailler ?")
    print(gameinfo.get_game_list_numbered())
    chosent_game_id = int(input(""))
    is_game_chosen = gameinfo.game_id_exists(chosent_game_id)
        
chosen_game_info = gameinfo.get_game_info(chosent_game_id)

print("----Vous avez choisi {}-----".format(chosen_game_info["displayed_name"]))



#--------Choix des paramètres---------

available_settings = chosen_game_info["available_settings"]
game_settings={}
if len(available_settings) > 0:
    print("-----Selection des paramètres------")
    game_settings = console_helper.enter_values(available_settings)
    

env = gym.make(chosen_game_info["name"], **game_settings)
is_env_observation_discrete = isinstance(env.observation_space, gym.spaces.Discrete)


#--------Choix de l'agent--------

print("-----Selection du type d'agent-------")
if is_env_observation_discrete:
    print("0) Q-Learning tabulaire")
print("1) Deep Q-Learning")
selected_agent_type_id = int(input(""))

if not is_env_observation_discrete:
    selected_agent_type_id=1    #Force DQN if env isn't discrete, as it's the only that support continuous state space

hyperparams_to_fill=[]
if selected_agent_type_id==0:
    hyperparams_to_fill=QLEARNING_HYPERPARAMS
else:
    hyperparams_to_fill=DQN_HYPERPARAMS

agent = None


print("----Chargement de l'agent----")

#list available files
suffix=""
if selected_agent_type_id==0:
    suffix=QLearningAgent.QLAGENT_FILE_SUFFIX
else:
    suffix=DQNAgent.DQN_FILE_SUFFIX
files = glob.glob(SAVE_FOLDER+"*"+suffix)

#list options (only display loading if possible)
print("0) Créer l'agent")
if len(files) > 0:
    print("1) Charger un fichier")
selected_loading_action = int(input(""))

if selected_loading_action==0 or len(files)==0:
    agents_hyper_parameters = {}

    print("-----Création de l'agent------\nVeuillez rentrer les hyperparametres")
    
    agents_hyper_parameters = console_helper.enter_values(hyperparams_to_fill)

    #Create the agent
    if selected_agent_type_id==0:   #QLearning
        agent = QLearningAgent(env.observation_space.n, env.action_space.n, **agents_hyper_parameters)
    elif selected_agent_type_id==1: #DQN
        input_size = 1
        if not is_env_observation_discrete:
            input_size = env.observation_space.shape[0]
        agent = DQNAgent(input_size, env.action_space.n, **agents_hyper_parameters)

else:
    for i in range(len(files)):
        print("{}) {}".format(i, os.path.basename(files[i])))
    selected_file_id = -1
    while not (selected_file_id>=0 and selected_file_id <len(files)):
        selected_file_id = int(input(""))
    
    selected_file_path = files[selected_file_id]
    if selected_agent_type_id==0:
        agent = QLearningAgent.load(selected_file_path)
    elif selected_agent_type_id==1:
        agent = DQNAgent.load(selected_file_path)



#Agent is now loaded or create, and env is selected and loaded

#----------Main menu-------------
stop = False
while not stop:
    print("------Menu principal-------")
    displayed_options = "1) Entrainer l'agent\n2) Tester l'agent\n3) Sauvegarder l'agent\n4) Voir un résumé de l'agent\n0) Quitter"
    if isinstance(agent, DQNAgent):
        displayed_options += "\n5) Vider l'Experience Replay"
    print(displayed_options)
    choice=int(input(""))

    if choice==0:
        stop = True
    elif choice==1:
        training_params={}
        if isinstance(agent, DQNAgent):
            training_params = console_helper.enter_values(DQN_TRAINPARAMS)
            if chosen_game_info["name"] in ENVS_DQN_POSTPROCESS_TRAINPARAMS:
                training_params.update(ENVS_DQN_POSTPROCESS_TRAINPARAMS[chosen_game_info["name"]])
        elif isinstance(agent, QLearningAgent):
            training_params = console_helper.enter_values(QLEARNING_TRAINPARAMS)
        training_params["train"]=True
        agent.train(env, **training_params)
    elif choice==2:
        training_params={}
        if isinstance(agent, DQNAgent): #If is a DQN Agent
            training_params = console_helper.enter_values(DQN_TESTPARAMS)
            #If the game has special post / pre process functions, add them to the params
            if chosen_game_info["name"] in ENVS_DQN_POSTPROCESS_TRAINPARAMS:
                training_params.update(ENVS_DQN_POSTPROCESS_TRAINPARAMS[chosen_game_info["name"]])
        elif isinstance(agent, QLearningAgent): #If is a Q Learning agent
            training_params = console_helper.enter_values(QLEARNING_TESTPARAMS)
        training_params["train"]=False
        training_params["epsilon_decay"]=1.0

        #start test
        steps_per_episode_history, rewards_history, _ = agent.train(env, **training_params)

        #display results
        test_str = "-------Resultats du test-------"
        test_str += "\n- nombre d'épisodes : {}".format(training_params["episode_count"])
        test_str += "\n- epsilon : {}".format(training_params["epsilon"])
        test_str += "\n- nombre d'étapes maximal : {}".format(np.max(steps_per_episode_history))
        test_str += "\n- nombre d'étapes minimal : {}".format(np.min(steps_per_episode_history))
        test_str += "\n- nombre d'étapes moyen : {}".format(np.average(steps_per_episode_history))
        test_str += "\n- récompense maximale : {}".format(np.max(rewards_history))
        test_str += "\n- récompense minimale : {}".format(np.min(rewards_history))
        test_str += "\n- récompense moyenne : {}".format(np.average(rewards_history))
        print(test_str)

    elif choice==3:
        print("-------Sauvegarde--------")
        filename = SAVE_FOLDER + input("Choisissez le nom du fichier de sauvegarde de votre agent\n")
        agent.save(filename)
    elif choice==4:
        print("------Résumé------")
        print(agent.get_written_summary())
    elif isinstance(agent, DQNAgent):
        if choice == 5:
            print("------Vidage de l'Experience Replay------")
            expreplay_size = len(agent.memory)
            agent.memory.clear()
            print("L'Experience Replay d'une taille de {} transitions a été vidé !".format(expreplay_size))