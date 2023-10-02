## TO DO ##
# add option to alter/set termination reward
# add option to alter step reward
# add checkbox option to use RBF kernels
# add checkbox option to use standard scaler
# remove input dimension option after adding auto detect input dims to RL agent code

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
    st.session_state.ax1.plot(st.session_state.avg_rewards, label='Average Reward')
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

def draw_plots(fig, ax):
    # Plot rewards and avg rewards on the first subplot
    ax[0].plot(st.session_state.rewards, label='Reward')
    ax[0].plot(st.session_state.avg_rewards, label='Average Reward')
    ax[0].set_title('Rewards over Episodes')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Rewards')
    ax[0].legend()
    ax[0].grid(True)

    # Plot the last points with values
    if st.session_state.rewards:
        ax[0].text(len(st.session_state.rewards)-1, st.session_state.rewards[-1], str(st.session_state.rewards[-1]), fontsize=9)
    if st.session_state.avg_rewards:
        ax[0].text(len(st.session_state.avg_rewards)-1, st.session_state.avg_rewards[-1], str(st.session_state.avg_rewards[-1]), fontsize=9)

    # Plot epsilon and learning rate on the second subplot
    ax[1].plot(st.session_state.epsilons, label='Epsilon', color='g')
    ax[1].plot(st.session_state.learning_rates, label='Learning Rate', color='r')
    ax[1].set_title('Epsilon and Learning Rate over Episodes')
    ax[1].set_xlabel('Episodes')
    ax[1].set_ylabel('Value')
    ax[1].legend()
    ax[1].grid(True)

    # Plot the last points with values
    if st.session_state.epsilons:
        ax[1].text(len(st.session_state.epsilons)-1, st.session_state.epsilons[-1], f'{st.session_state.epsilons[-1]:.2f}', fontsize=9)
    if st.session_state.learning_rates:
        ax[1].text(len(st.session_state.learning_rates)-1, st.session_state.learning_rates[-1], f'{st.session_state.learning_rates[-1]:.2f}', fontsize=9)

    st.pyplot(fig)

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
# set number of kernels
st.sidebar.header('Radial Basis Function Kernels')
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
    input_dim = st.sidebar.number_input(f'Input dimension for layer {i+1}', min_value=1, value=1, step=1)
    output_dim = st.sidebar.number_input(f'Output dimension for layer {i+1}', min_value=1, value=1, step=1)
    layer_params.append((input_dim, output_dim))

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

# set gamma
st.sidebar.subheader('Gamma')
gamma = st.sidebar.number_input('Gamma (Return Decay)', min_value=0.0, max_value=1.0, value=0.97, step=0.01)

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
    env = gra.gym.make('CartPole-v1')

    # collect sample states to fit scaler and rbf kernels
    sample_states = gra.get_sample_states(env, 10000)

    # instantiate scaler
    scaler = gra.StandardScaler()

    # instantiate rbf kernels and merge into a feature union
    sigmas, n_components = zip(*kernels_params)
    rbf_samplers = gra.create_rbf_samplers(sigmas, n_components)
    # create a FeatureUnion object of RBFSamplers
    rbf_union = gra.FeatureUnion([*rbf_samplers])

    # instantiate a transformer class
    transformer = gra.Transformer()
    # add scaler and rbf union
    transformer.add(scaler)
    transformer.add(rbf_union)
    # fit transformer to sample states
    transformer.fit(sample_states)

    # create the layers of the model
    input_dims, output_dims = zip(*layer_params)
    layers = []
    for i, o in zip(input_dims, output_dims):
        layers.append(gra.DenseLayer(i,o))
    
    loss = gra.MeanSquaredError()

    # create a model object
    model = gra.Model(layers=layers,
                loss=loss,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay,
                decay_rate=decay_rate,
                min_learning_rate=min_learning_rate)

    # create agent
    agent = gra.Agent(env, model, transformer, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

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

