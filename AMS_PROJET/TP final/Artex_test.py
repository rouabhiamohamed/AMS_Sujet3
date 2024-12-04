from artext import Artext

artxt = Artext()
artxt.samples = 5
artxt.error_rate = 0.25
sent = 'This is a sample sentence to be noised.'
noises = artxt.noise_sentence(sent)
print(noises)