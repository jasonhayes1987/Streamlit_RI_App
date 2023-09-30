import streamlit as st
import mountain_car_rbf as mcr
import matplotlib.pyplot as plt


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
learning_rate_decay = st.sidebar.selectbox('Learning Rate Decay', ['Constant','Linear','Exponential','Inverse'])
decay_rate = st.sidebar.number_input('Decay Rate', min_value=0.0, max_value=1.0, step=0.01, value=0.1)
min_learning_rate = st.sidebar.number_input('Minimum Learning Rate', min_value=0.0, max_value=1.0, step=0.001, value=0.001, format='%.3f')

# set epsilon
st.sidebar.header('Epsilon')
epsilon = st.sidebar.number_input('Epsilon', min_value=0.0, max_value=1.0, value=0.1, step=0.1)
min_epsilon = st.sidebar.number_input('Minimum Epsilon', min_value=0.0, max_value=1.0, step=0.1, value=0.1)
epsilon_decay = st.sidebar.number_input('Epsilon Decay', min_value=0.0, max_value=1.0, step=0.01, value=0.97)

# set gamma
st.sidebar.subheader('Gamma')
gamma = st.sidebar.number_input('Gamma (Return Decay)', min_value=0.0, max_value=1.0, value=0.97, step=0.01)

# set number of training episodes
episodes = st.sidebar.number_input('Training Episodes', min_value=1, value=1000, step=1)

ready_for_training = st.button('Train')

if ready_for_training:
    #instantiate cartpole env
    env = mcr.gym.make('CartPole-v1')

    # collect sample states to fit scaler and rbf kernels
    sample_states = mcr.get_sample_states(env, 10000)

    # instantiate scaler
    scaler = mcr.StandardScaler()

    # instantiate rbf kernels and merge into a feature union
    sigmas, n_components = zip(*kernels_params)
    rbf_samplers = mcr.create_rbf_samplers(sigmas, n_components)
    # create a FeatureUnion object of RBFSamplers
    rbf_union = mcr.FeatureUnion([*rbf_samplers])

    # instantiate a transformer class
    transformer = mcr.Transformer()
    # add scaler and rbf union
    transformer.add(scaler)
    transformer.add(rbf_union)
    # fit transformer to sample states
    transformer.fit(sample_states)

    # create the layers of the model
    input_dims, output_dims = zip(*layer_params)
    layers = []
    for i, o in zip(input_dims, output_dims):
        layers.append(mcr.DenseLayer(i,o))
    
    loss = mcr.MeanSquaredError()

    # create a model object
    model = mcr.Model(layers=layers,
                loss=loss,
                learning_rate=learning_rate,
                learning_rate_decay=learning_rate_decay,
                decay_rate=decay_rate,
                min_learning_rate=min_learning_rate)

    # create agent
    agent = mcr.Agent(env, model, transformer, epsilon=epsilon, min_epsilon=min_epsilon, epsilon_decay=epsilon_decay, gamma=gamma)

    # placeholders for the plots
    reward_placeholder = st.empty()
    avg_reward_placeholder = st.empty()
    epsilon_placeholder = st.empty()
    learning_rate_placeholder = st.empty()

    rewards = []
    avg_rewards = []
    epsilons = []
    learning_rates = []

    with st.empty():
        for status in agent.train(episodes, streamlit=True, update_freq=10):
            episode, reward, avg_reward, epsilon, learning_rate = status
            rewards.append(reward)
            avg_rewards.append(avg_reward)
            epsilons.append(epsilon)
            learning_rates.append(learning_rate)

            # Create a figure and axes
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))

            # Plot rewards over episodes
            ax[0, 0].plot(rewards)
            ax[0, 0].set_title('Rewards over Episodes')
            ax[0, 0].set_xlabel('Episodes')
            ax[0, 0].set_ylabel('Rewards')

            # Plot average rewards over the last 100 episodes
            ax[0, 1].plot(avg_rewards)
            ax[0, 1].set_title('Average Rewards over Last 100 Episodes')
            ax[0, 1].set_xlabel('Episodes')
            ax[0, 1].set_ylabel('Average Rewards')

            # Plot epsilon over episodes
            ax[1, 0].plot(epsilons)
            ax[1, 0].set_title('Epsilon over Episodes')
            ax[1, 0].set_xlabel('Episodes')
            ax[1, 0].set_ylabel('Epsilon')

            # Plot learning rate over episodes
            ax[1, 1].plot(learning_rates)
            ax[1, 1].set_title('Learning Rate over Episodes')
            ax[1, 1].set_xlabel('Episodes')
            ax[1, 1].set_ylabel('Learning Rate')

            # Display the plots
            st.pyplot(fig)
