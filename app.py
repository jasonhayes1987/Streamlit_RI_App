

import streamlit as st
import gym_rl_agent as gra
import matplotlib.pyplot as plt

# Function to initialize the plots
def init_plots():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    ax1.set_title('Rewards and Average Rewards')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Reward Value')
    ax2.set_title('Epsilon and Learning Rate')
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel('Value')

    return fig, ax1, ax2

# Function to update the plots
def update_plots(plot_container):
    st.session_state.ax1.clear()
    st.session_state.ax2.clear()

    # Plot rewards and avg rewards on the first subplot
    st.session_state.ax1.plot(st.session_state.rewards, label='Reward')
    st.session_state.ax1.plot(st.session_state.avg_rewards, label='Average Reward/100 Episodes')
    st.session_state.ax1.set_title('Rewards over Episodes')
    st.session_state.ax1.set_xlabel('Episodes')
    st.session_state.ax1.set_ylabel('Rewards')
    st.session_state.ax1.legend()
    st.session_state.ax1.grid(True)

    # Plot the last points with values
    if st.session_state.rewards:
        st.session_state.ax1.text(len(st.session_state.rewards)-1, st.session_state.rewards[-1], str(st.session_state.rewards[-1]), fontsize=9)
    if st.session_state.avg_rewards:
        st.session_state.ax1.text(len(st.session_state.avg_rewards)-1, st.session_state.avg_rewards[-1], str(st.session_state.avg_rewards[-1]), fontsize=9)

    # Plot epsilon and learning rate on the second subplot
    st.session_state.ax2.plot(st.session_state.epsilons, label='Epsilon', color='g')
    st.session_state.ax2.plot(st.session_state.learning_rates, label='Learning Rate', color='r')
    st.session_state.ax2.set_title('Epsilon and Learning Rate over Episodes')
    st.session_state.ax2.set_xlabel('Episodes')
    st.session_state.ax2.set_ylabel('Value')
    st.session_state.ax2.legend()
    st.session_state.ax2.grid(True)

    # Plot the last points with values
    if st.session_state.epsilons:
        st.session_state.ax2.text(len(st.session_state.epsilons)-1, st.session_state.epsilons[-1], f'{st.session_state.epsilons[-1]:.2f}', fontsize=9)
    if st.session_state.learning_rates:
        st.session_state.ax2.text(len(st.session_state.learning_rates)-1, st.session_state.learning_rates[-1], f'{st.session_state.learning_rates[-1]:.2f}', fontsize=9)

    # Refresh the plot in Streamlit
    plot_container.pyplot(st.session_state.figure)

# Initialize session state variables if they haven't been already
if 'agent_trained' not in st.session_state:
    st.session_state['agent_trained'] = False
if 'agent' not in st.session_state:
    st.session_state['agent'] = None
if 'rewards' not in st.session_state:
    st.session_state['rewards'] = []
if 'avg_rewards' not in st.session_state:
    st.session_state['avg_rewards'] = []
if 'epsilons' not in st.session_state:
    st.session_state['epsilons'] = []
if 'learning_rates' not in st.session_state:
    st.session_state['learning_rates'] = []
if 'figure' not in st.session_state:
    st.session_state.figure, st.session_state.ax1, st.session_state.ax2 = init_plots()


# set title of page
st.title('Cartpole RL Agent')

## set parameters  ##
gym_env = st.sidebar.selectbox('Gym Environment', ["Pendulum-v1","Acrobot-v1","CartPole-v1","MountainCarContinuous-v0","MountainCar-v0"])
# option to scale data
st.sidebar.header('Data Transformers')
use_scaler = st.sidebar.checkbox('Use Standard Scaler')
# option to use RBF kernels
use_rbf = st.sidebar.checkbox('Use RBF Kernels')
if use_rbf:
    # set number of kernels
    st.sidebar.subheader('Radial Basis Function Kernels')
    num_kernels = st.sidebar.number_input('Number of RBF kernels', min_value=1, max_value=10, value=4)
    kernels_params = []

    for i in range(num_kernels):
        st.sidebar.subheader(f'Parameters for RBF Kernel {i+1}')
        variance = st.sidebar.number_input(f'Variance for RBF Kernel {i+1}', min_value=0.01, max_value=5.0, value=0.5, step=0.1)
        num_centers = st.sidebar.number_input(f'Number of centers for RBF Kernel {i+1}', min_value=10, max_value=2000, value=1000, step=10)
        kernels_params.append((variance, num_centers))

# set number of layers in model
st.sidebar.header('Dense Layers')
num_layers = st.sidebar.number_input('Number of layers in network', min_value=1, max_value=10, value=1)
layer_params = []

for i in range(num_layers):
    st.sidebar.subheader(f'Parameters for layer {i+1}')
    # input_dim = st.sidebar.number_input(f'Input dimension for layer {i+1}', min_value=1, value=1, step=1)
    output_dim = st.sidebar.number_input(f'Output dimension for layer {i+1}', min_value=1, value=1, step=1)
    layer_params.append(output_dim)

# set learning rate parameters
st.sidebar.header('Learning Rate')
learning_rate = st.sidebar.number_input('Learning Rate', min_value=0.001, max_value=1.0, step=0.001, value=0.001, format='%.3f')
learning_rate_decay = st.sidebar.selectbox('Learning Rate Decay', ['constant','linear','exponential','inverse'])
decay_rate = st.sidebar.number_input('Decay Rate', min_value=0.0, max_value=1.0, step=0.01, value=0.1)
min_learning_rate = st.sidebar.number_input('Minimum Learning Rate', min_value=0.0, max_value=1.0, step=0.001, value=0.001, format='%.3f')

# set epsilon
st.sidebar.header('Epsilon')
epsilon = st.sidebar.number_input('Epsilon', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
min_epsilon = st.sidebar.number_input('Minimum Epsilon', min_value=0.0, max_value=1.0, step=0.01, value=0.1)
epsilon_decay = st.sidebar.number_input('Epsilon Decay', min_value=0.0, max_value=1.0, step=0.01, value=0.97)

# set TD Lambda parameters
st.sidebar.subheader('TD Lambda')
gamma = st.sidebar.number_input('Gamma (Return Decay)', min_value=0.0, max_value=1.0, value=0.97, step=0.01)
lmbda = st.sidebar.number_input('Lambda (Previous Gradient Decay)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.sidebar.header('Training')
# step reward adjustment selection checkbox
if st.sidebar.checkbox('Adjust Step Reward'):
    # step reward adjustment
    r_step = st.sidebar.number_input(f'Step Reward Adjustment', value=0.0, step=1.0)
else:
    r_step = None
# termination reward adjustment selection box
if st.sidebar.checkbox('Adjust Termination Reward'):
    # termination reward adjustment
    r_term = st.sidebar.number_input(f'Termination Reward Adjustment', value=0.0, step=1.0)
else:
    r_term = None
# set number of training episodes
episodes = st.sidebar.number_input('Training Episodes', min_value=1, value=1000, step=1)

col1, col2, col3 = st.columns(3)
with col1:
    live_update = st.checkbox('Live Update')

with col2:
    train_button = st.button('Train')

# create a container for plotting data
plot_container = st.empty()
if train_button:
    # Clear previous training data from session state
    st.session_state.rewards = []
    st.session_state.avg_rewards = []
    st.session_state.epsilons = []
    st.session_state.learning_rates = []

    #instantiate cartpole env
    env = gra.gym.make(gym_env)

    # collect sample states to fit scaler and rbf kernels
    sample_states = gra.get_sample_states(env, 1000)  

    if use_scaler or use_rbf:
        # instantiate transformer
        transformer = gra.Transformer()
        # add scaler if set to use
        if use_scaler:
            # instantiate scaler
            scaler = gra.StandardScaler()
            transformer.add(scaler)
        # add rbf if set to use
        if use_rbf:
            # instantiate rbf kernels and merge into a feature union
            sigmas, n_components = zip(*kernels_params)
            rbf_samplers = gra.create_rbf_samplers(sigmas, n_components)
            # create a FeatureUnion object of RBFSamplers
            rbf_union = gra.FeatureUnion([*rbf_samplers])
            transformer.add(rbf_union)
        # fit transformer to sample states
        transformer.fit(sample_states)
        # transform sample states to compile model to correct input data shape
        sample_states = transformer.transform(sample_states)
    else:
        transformer = None
    
    

    # create the layers of the model
    layers = []
    for i in layer_params:
        layers.append(gra.DenseLayer(i))
    
    loss = gra.TDError(gamma=gamma, lmbda=lmbda)

    # create a model object
    model = gra.Model(layers=layers,
                loss=loss,
                input_data=sample_states,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay,
                decay_rate=decay_rate,
                min_learning_rate=min_learning_rate)

    # create agent
    agent = gra.Agent(env,
                      model,
                      transformer,
                      epsilon=epsilon,
                      min_epsilon=min_epsilon,
                      epsilon_decay=epsilon_decay,
                      gamma=gamma,
                      step_reward=r_step,
                      term_reward=r_term)

    # placeholders for the plots
    # reward_placeholder = st.empty()
    # avg_reward_placeholder = st.empty()
    # epsilon_placeholder = st.empty()
    # learning_rate_placeholder = st.empty()
    with st.empty():
        for status in agent.train(episodes, streamlit=True, update_freq=1):
            episode, reward, avg_reward, epsilon, learning_rate = status.values()
            st.session_state.rewards.append(reward)
            st.session_state.avg_rewards.append(avg_reward)
            st.session_state.epsilons.append(epsilon)
            st.session_state.learning_rates.append(learning_rate)
            if live_update:
                update_plots(plot_container)
        else:
            update_plots(plot_container)

    st.session_state.agent_trained = True
    st.session_state.agent = agent
    
update_plots(plot_container)
if st.session_state.agent_trained:
    with col3:
        play_button = st.button('Play')
        if play_button:
            with st.spinner('Playing...'):
                rewards = st.session_state.agent.play(n_episodes=1, render=True)
            st.write('Total Rewards:', sum(rewards))

