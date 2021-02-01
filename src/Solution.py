import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy import signal
from random import randint


"""
1. Nahrajte dvˇe testovac´ı nahr´avky — jeden a ten sam´y t´on (napˇr´ıklad samohl´aska “´a”) bez rouˇsky
a s rouˇskou. Nahr´avku bez rouˇsky pojmenujte maskoff tone.wav a s nasazenou rouˇskou maskon tone.wav
"""
fsMaskonTone, maskonTone = wavfile.read('../audio/maskon_tone.wav')
lenghtMaskonTone = maskonTone.size
print("1:")
print("Delka v [s] maskon_tone.wav =", lenghtMaskonTone/fsMaskonTone)
print("Delka ve vzorcich maskon_tone.wav =", lenghtMaskonTone)
print("Frenkvence v [Hz] maskon_tone.wav =", fsMaskonTone, end = '\n\n')

fsMaskoffTone, maskoffTone = wavfile.read('../audio/maskoff_tone.wav')
lenghtMaskoffTone = maskoffTone.size
print("Delka v [s] maskoff_tone.wav =", lenghtMaskoffTone/fsMaskoffTone)
print("Delka ve vzorcich maskoff_tone.wav =", lenghtMaskoffTone)
print("Frenkvence v [Hz] maskoff_tone.wav =", fsMaskoffTone, end = '\n\n')

"""
2.Nahrajte jednu a tu samou vˇetu bez rouˇsky a s rouˇskou. Nahr´avku bez rouˇsky pojmenujte
maskoff sentence.wav a s nasazenou rouˇskou maskon sentence.wav
"""

print("2:")
fsMaskonSent, MaskonSent = wavfile.read('../audio/maskon_sentence.wav')
lenghtMaskonSent = MaskonSent.size
print("Delka v [s] maskon_sentence.wav =", lenghtMaskonSent/fsMaskonSent)
print("Delka ve vzorcich maskon_sentence.wav =", lenghtMaskonSent)
print("Frenkvence v [Hz] maskon_sentence.wav =", fsMaskonSent, end='\n\n')


fsMaskoffSent, MaskoffSent = wavfile.read('../audio/maskoff_sentence.wav')
lenghtMaskoffSent = MaskoffSent.size
print("Delka v [s] maskoff_sentence.wav =", lenghtMaskoffSent/fsMaskoffSent)
print("Delka ve vzorcich maskoff_sentence.wav =", lenghtMaskoffSent)
print("Frenkvence v [Hz] maskoff_sentence.wav =", fsMaskoffSent, end='\n\n')

"""
3.Z kaˇzd´e testovac´ı nahr´avky (t´on˚u) extrahujte ˇc´ast o d´elce 1 sekunda, ve kter´e se budou sign´aly co
nejm´enˇe liˇsit. Nyn´ı oba extrahovan´e ´useky rozdˇelte na r´amce. R´amec je ˇc´ast ˇreˇcov´eho sign´alu o d´elce typicky 20 – 25 ms.
Pouˇzijte 20 ms. Jednotliv´e r´amce se pˇrekr´yvaj´ı o 10 ms. Na ´useku dlouh´em 1 sekundu byste tedy mˇeli z´ıskat
100 r´amc˚u.
"""

maskonTone = maskonTone[4000:20000]
maskoffTone = maskoffTone[4000:20000]

maskonTone = maskonTone - np.mean(maskonTone) 
maskonTone = maskonTone / np.abs(maskonTone).max()

maskoffTone = maskoffTone - np.mean(maskoffTone)
maskoffTone = maskoffTone / np.abs(maskoffTone).max()

i = 0
frameMaskOnTone = np.empty((99, 320), float)
frameMaskOffTone = np.empty((99, 320), float)
indexTone = 0
for index in range(0, 99):
    for index2 in range(0, 320):
        frameMaskOffTone[index][index2] = maskoffTone[indexTone]
        frameMaskOnTone[index][index2] = maskonTone[indexTone]
        indexTone += 1
    indexTone -= 160

plt.figure()
plt.plot(np.arange(0.000, 0.02, 0.0000625), frameMaskOffTone[98])
plt.gca().set_title('Čast sígnálu maskoff_tone.wav o delce 20ms(1 ramec)')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('y')
plt.savefig('3.1.png')
plt.figure()
plt.plot(np.arange(0.0, 0.02, 0.0000625), frameMaskOnTone[98])
plt.gca().set_title('Čast sígnálu maskon_tone.wav o delce 20ms(1 ramec)')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('y')
plt.savefig('3.2.png')


"""
4.Pokud jste nahr´avali testovac´ı t´ony ve stejn´em prostˇred´ı a se stejn´ym zaˇr´ızen´ım, jedin´y rozd´ıl,
kter´y by mohl b´yt mezi nahr´avkami (kromˇe rouˇsky), je v´yˇska t´onu. Zkontrolujeme tedy, jestli se neliˇs´ı, to
by n´am totiˇz mohlo dˇelat pot´ıˇze pˇri odhadu filtru.
"""
print("4.")

helperFrameMaskOffTone = np.empty((99,320), float)
index = 0
index1 = 0
for a in frameMaskOffTone:
    maximum = np.max(a)
    for x in a:
        if x > abs(maximum) * 0.7:
            helperFrameMaskOffTone[index][index1] = 1
        elif x < abs(maximum) * -0.7:
            helperFrameMaskOffTone[index][index1] = -1
        else:
            helperFrameMaskOffTone[index][index1] = 0
        index1 += 1
    index += 1
    index1 = 0

helperFrameMaskOnTone = np.empty((99,320), float)
index = 0
index1 = 0
for a in frameMaskOnTone:
    maximum = np.max(a)
    for x in a:
        if x > abs(maximum) * 0.7:
            helperFrameMaskOnTone[index][index1] = 1
        elif x < abs(maximum) * -0.7:
            helperFrameMaskOnTone[index][index1] = -1
        else:
            helperFrameMaskOnTone[index][index1] = 0
        index1 += 1
    index += 1
    index1 = 0

autoCorrMaskOn = np.empty((99, 320), float)
autoCorrMaskOff = np.empty((99, 320), float)

step = 0
i = 0
koefOn = 0
koefOff = 0
for i in range(99):
    step = 0
    index = 0
    while step < 319:
        koefOff += helperFrameMaskOffTone[i][index] * helperFrameMaskOffTone[i][index+step]
        koefOn += helperFrameMaskOnTone[i][index] * helperFrameMaskOnTone[i][index+step]
        index += 1
        if (index + step) > 319:
            autoCorrMaskOff[i][step] = koefOff
            autoCorrMaskOn[i][step] = koefOn
            koefOff = 0
            koefOn = 0
            index = 0
            step += 1

f0MaskOn = []
f0MaskOff = []

i = 0

for i in range(99):
    resOff = np.where(autoCorrMaskOff[i][15:] == max(autoCorrMaskOff[i][15:]))
    resOn = np.where(autoCorrMaskOn[i][15:] == max(autoCorrMaskOn[i][15:]))
    MaskOnIndx = 16000 / (resOn[0][0] + 15)
    MaskOffIndx = 16000 / (resOff[0][0] + 15)

    f0MaskOn.append(MaskOnIndx)
    f0MaskOff.append(MaskOffIndx)
    

plt.figure(figsize=(10,4))
plt.plot(np.arange(0.000, 0.02, 0.0000625), frameMaskOffTone[33])
plt.gca().set_title('Rámec')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('y')
plt.savefig('4.1.png')

plt.figure(figsize=(10,4))
plt.plot(f0MaskOff, label = "bez roušky")
plt.plot(f0MaskOn, label = "s rouškou")
plt.legend()
plt.gca().set_title('Základní frekvence ramců')
plt.gca().set_xlabel('ramce')
plt.gca().set_ylabel('f0')
plt.savefig('4.4.png')

plt.figure(figsize=(10,4))
plt.plot(autoCorrMaskOff[33])
plt.gca().set_title('Autokorelace')
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('y')
plt.axvline(x = 15, color = 'red')
res = np.where(autoCorrMaskOff[33][15:] == max(autoCorrMaskOff[33][15:]))
plt.axvline(x = res[0][0]+15, color = 'grey', ymax = max(autoCorrMaskOff[33][15:])/25)
plt.text(16, 0,'Práh',rotation=90, color = 'red')
plt.text(res[0][0]+15, 0, 'Lag', rotation=90, color = 'grey')
plt.savefig('4.3.png')

plt.figure(figsize=(10,4))
plt.plot(np.arange(0.000, 0.02, 0.0000625), helperFrameMaskOffTone[33])
plt.gca().set_title('Centrální klipování s 70%')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('y')
plt.savefig('4.2.png')


print("Pruměr základní frekvencí rámců bez roušky = ", np.mean(f0MaskOff))
print("Pruměr základní frekvencí rámců s rouškou = ", np.mean(f0MaskOn))
print("Rozptyl frekvencí rámců bez roušky = ", np.var(f0MaskOff))
print("Rozptyl frekvencí rámců s rouškou = ", np.var(f0MaskOn), end="\n\n")

"""
5.Nyn´ı uˇz m´ame dvˇe sady r´amc˚u s ˇreˇc´ı, u kter´ych pˇredpokl´ad´ame, ˇze jedin´ym rozd´ılem je ”rouˇskov´y
filtr”. Pod´ıvejme se na nˇe ve spektr´aln´ı oblasti
"""
print("5.")

def dft(s):
    s = np.asarray(s, dtype=float)
    N = s.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n/N)
    return np.dot(M, s)

MaskOffDft = np.zeros(shape = (99, 704))
MaskOnDft = np.zeros(shape = (99, 704))

MaskOffDft = np.append(frameMaskOffTone, MaskOffDft, axis = 1)
MaskOnDft = np.append(frameMaskOnTone, MaskOnDft, axis = 1)
randomInt = randint(0, 98)
CheckFunction = dft(MaskOffDft[randomInt])
MaskOffDft = np.apply_along_axis(np.fft.fft, 1, MaskOffDft)
MaskOnDft = np.apply_along_axis(np.fft.fft, 1, MaskOnDft)
MaskOn11 = MaskOnDft[32]
print("Compare np.fft.fft(x) and dft(x):", np.allclose(MaskOffDft[randomInt], CheckFunction))
PMaskOff = 10 * np.log10(np.abs(MaskOffDft[:, 0:512])**2)
PMaskOn = 10 * np.log10(np.abs(MaskOnDft[:, 0:512])**2)


plt.figure(figsize=(9,4))
plt.imshow(np.transpose(PMaskOff) , extent=[0, 1, 0, 8000], aspect="auto", origin="lower")
plt.gca().set_title('Spektrogram bez roušky')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.savefig("5.1.png")


plt.figure(figsize=(9,4))
plt.imshow(np.transpose(PMaskOn) , extent=[0, 1, 0, 8000], aspect="auto", origin="lower")
plt.gca().set_title('Spektrogram s rouškou')
plt.gca().set_xlabel('Čas [s]')
plt.gca().set_ylabel('Frekvence [Hz]')
cbar = plt.colorbar()
cbar.set_label('Spektralní hustota výkonu [dB]', rotation=270, labelpad=15)

plt.savefig("5.2.png")

"""
6. Kdyˇz m´ame k disposici spektrogramy sign´al˚u s rouˇskou a bez, m˚uˇzeme spoˇc´ıtat frekvenˇcn´ı charakteristiku rouˇsky. Tu z´ısk´ame (v´ypoˇctem, kter´y z´amˇernˇe nen´ı uveden) pro kaˇzd´y r´amec, zpr˚umˇerujeme ji pˇres
vˇsechny r´amce, abychom z´ıskali jednu frekvenˇcn´ı charakteristiku. Velmi peˇclivˇe uvaˇzte, jak budete prov´adˇet
pod´ıl a pr˚umˇerov´an´ı komplexn´ıch ˇc´ısel. Velmi doporuˇcujeme pr˚umˇerovat pouze absolutn´ı hodnoty.
"""

frqChar = abs(MaskOnDft[:, 0:512]) / abs(MaskOffDft[:, 0:512])
frqChar = np.mean(frqChar, axis=0)

plt.figure()

plt.plot(frqChar)
plt.gca().set_title("Frekvenční charakteristika")
plt.gca().set_xlabel("Vzorky")
plt.gca().set_ylabel('y')

plt.savefig("6.1.png")

"""
7.Samotn´a filtrace bude jednoduˇsˇs´ı v ˇcasov´e oblasti, frekvenˇcn´ı charakteristiku pˇrevedeme na 
impulsn´ı odezvu pomoc´ı inverzn´ı DFT. Opˇet naimplementujte vlastn´ı verzi a porovnejte s 
knihovn´ı implementac´ı (napˇr. np.fft.ifft).
"""
print("\n7.")
def idft(s):
    s = np.asarray(s, dtype=float)
    N = s.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(2j * np.pi * k * n/N)
    return np.dot(M, s)/N

idftFrq = np.zeros(shape = (1, 512))
idftFrq = np.append(frqChar, idftFrq)
print("Compare np.fft.ifft(x) and idft(x):", np.allclose(idft(idftFrq), np.fft.ifft(idftFrq)))
idftFrq = idft(idftFrq)

plt.figure()
plt.plot(abs(idftFrq[0:512]))
plt.gca().set_title("Impulsní odezva")
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('y')
plt.savefig("7.1.png")

"""
8.Kdyˇz uˇz m´ate koeficienty masky, m˚uˇzete jednoduˇse prov´est simulaci rouˇsky, tedy filtraci na
t´onu a na namluven´e vˇetˇe bez rouˇsky. Jak ji provedete, je ˇciˇste na V´as, ale m˚uˇzete zkusit pouˇz´ıt funkci
scipy.signal.lfilter.
"""

y1 = signal.lfilter(abs(idftFrq[:512]), [1], MaskoffSent) / 10

#wavfile.write('sim_maskon_sentence.wav', 16000, y1.astype(np.int16))


_, maskoffTone = wavfile.read('../audio/maskoff_tone.wav')
y2 = signal.lfilter(abs(idftFrq[:512]), [1], maskoffTone) / 10

#wavfile.write('sim_maskon_tone.wav', 16000, y2.astype(np.int16))

plt.figure()
plt.plot(MaskoffSent)
plt.gca().set_title("Graf nahrané věty bez roušky")
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('y')
plt.savefig("8.1.png")

plt.figure()
plt.plot(MaskonSent)
plt.gca().set_title("Graf nahrané věty s rouškou")
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('y')
plt.savefig("8.2.png")

plt.figure()
plt.plot(y1)
plt.gca().set_title("Graf nahrané věty s simulovanou rouškou")
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('y')
plt.savefig("8.3.png")


"""
11.
"""

window = np.hamming(1024)

plt.figure()
plt.plot(window)
plt.gca().set_title('"Hamming Window"')
plt.gca().set_xlabel('Vzorky')
plt.gca().set_ylabel('Amplituda')
plt.savefig("11.1.png")

FreqRes = np.fft.fft(window, 1050) / 512
mag = np.abs(np.fft.fftshift(FreqRes))
freq = np.linspace(0, 1.0, len(FreqRes))
res = 20 * np.log10(mag)
res = np.clip(res, -100, 100)
plt.figure()
plt.plot(freq,res)
plt.gca().set_title('Spektrální oblast')
plt.gca().set_xlabel('Normalizovaná frekvence')
plt.gca().set_ylabel('Velikost[dB]')
plt.savefig("11.2.png")

MaskOffDft = np.zeros(shape = (99, 704))
MaskOnDft = np.zeros(shape = (99, 704))

MaskOffDft = np.append(frameMaskOffTone, MaskOffDft, axis = 1)
MaskOnDft = np.append(frameMaskOnTone, MaskOnDft, axis = 1)

hammingOff = np.multiply(window, MaskOffDft)
hammingOn = np.multiply(window, MaskOnDft)

hammingOff = np.apply_along_axis(np.fft.fft, 1, hammingOff)
hammingOn = np.apply_along_axis(np.fft.fft, 1, hammingOn)


plt.figure()
plt.plot(abs(hammingOn[32][:512]), label = "Hamming")
plt.plot(abs(MaskOn11[:512]), label = "Bez Hamming")
plt.legend()
plt.gca().set_title('Srovnání')
plt.gca().set_xlabel('vzorky')
plt.gca().set_ylabel('y')
plt.savefig("11.3.png")

frqChar = abs(hammingOn[:, 0:512]) / abs(hammingOff[:, 0:512])
frqChar = np.mean(frqChar, axis=0)

idftFrq = np.zeros(shape = (1, 512))
idftFrq = np.append(frqChar, idftFrq)
idftFrq = idft(idftFrq)

y1 = signal.lfilter(abs(idftFrq[:512]), [1], MaskoffSent) / 10

y2 = signal.lfilter(abs(idftFrq[:512]), [1], maskoffTone) / 10

#wavfile.write('sim_maskon_sentence_window.wav', 16000, y1.astype(np.int16))

#wavfile.write('sim_maskon_tone_window.wav', 16000, y2.astype(np.int16))