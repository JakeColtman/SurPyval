from SurPyval.symbolic import convert_to_symbols

def get_thetas(x_symbols, beta_symbols):
    multied = x_symbols.dot(beta_symbols.T)
    return np.apply_along_axis(lambda x: [exp(x[0])], 1, multied)

def individual_likihood(theatas, index):
    numerator = theatas[ii]
    denominator = sum(theatas[ii:])
    return numerator / denominator

def loss_function(theta_symbols, censored_mask):
    theatas = [x[0] for x in theta_symbols]
    total_loss = []
    for ii in range(len(theta_symbols)):
        if int(censored_mask[ii]) == 0:
            print("ignoring censored event")
            continue
        slice_start = ii 
        denom = sum(theatas[ii:])
        num = theatas[ii]
        loss_i = num / denom
        total_loss.append(loss_i)
    return reduce(lambda x, y: x * y, total_loss)

x_s = np.array(age).reshape(len(age), 1)
x_symbols = convert_to_symbols(x_s, "x")
beta_symbols = convert_to_symbols(np.array([[1]]), "beta")
theatas = get_thetas(x_symbols, beta_symbols)
total_loss = loss_function(theatas, arrest)