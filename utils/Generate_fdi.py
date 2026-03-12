import random


def generate_fdi_list(total_steps, num_attacks=10, min_faulty_temperature=150.0, max_faulty_temperature=160.0):
    fdi_list = []
    attack_steps = random.sample(range(total_steps), num_attacks)

    for step in attack_steps:
        # Generate a random faulty temperature within the specified range
        faulty_temperature = random.uniform(min_faulty_temperature, max_faulty_temperature)
        fdi_list.append((step, faulty_temperature))

    fdi_list.sort()
    return fdi_list


# Parameters
total_steps = 200
num_attacks = 10

# Generate the FDI list with adjusted faulty temperature values
fdi_list = generate_fdi_list(total_steps, num_attacks)
