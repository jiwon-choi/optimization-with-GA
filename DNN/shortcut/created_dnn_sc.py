
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist = input_data.read_data_sets("./MNIST_DATA", one_hot = True)
L0_sc = tf.placeholder('float', [None, 784])
Y = tf.placeholder('float', [None, 10])

W1 = tf.Variable(tf.zeros([784, 784]))
W2 = tf.Variable(tf.zeros([784, 784]))
W3 = tf.Variable(tf.zeros([784, 784]))
W4 = tf.Variable(tf.zeros([784, 784]))
W5 = tf.Variable(tf.zeros([784, 784]))
W6 = tf.Variable(tf.zeros([784, 784]))
W7 = tf.Variable(tf.zeros([784, 784]))
W8 = tf.Variable(tf.zeros([784, 784]))
W9 = tf.Variable(tf.zeros([784, 784]))
W10 = tf.Variable(tf.zeros([784, 784]))
W11 = tf.Variable(tf.zeros([784, 784]))
W12 = tf.Variable(tf.zeros([784, 784]))
W13 = tf.Variable(tf.zeros([784, 784]))
W14 = tf.Variable(tf.zeros([784, 784]))
W15 = tf.Variable(tf.zeros([784, 784]))
W16 = tf.Variable(tf.zeros([784, 784]))
W17 = tf.Variable(tf.zeros([784, 784]))
W18 = tf.Variable(tf.zeros([784, 784]))
W19 = tf.Variable(tf.zeros([784, 784]))
W20 = tf.Variable(tf.zeros([784, 784]))
W21 = tf.Variable(tf.zeros([784, 784]))
W22 = tf.Variable(tf.zeros([784, 784]))
W23 = tf.Variable(tf.zeros([784, 784]))
W24 = tf.Variable(tf.zeros([784, 784]))
W25 = tf.Variable(tf.zeros([784, 784]))
W26 = tf.Variable(tf.zeros([784, 784]))
W27 = tf.Variable(tf.zeros([784, 784]))
W28 = tf.Variable(tf.zeros([784, 784]))
W29 = tf.Variable(tf.zeros([784, 784]))
W30 = tf.Variable(tf.zeros([784, 784]))
W31 = tf.Variable(tf.zeros([784, 784]))
W32 = tf.Variable(tf.zeros([784, 784]))
W33 = tf.Variable(tf.zeros([784, 784]))
W34 = tf.Variable(tf.zeros([784, 784]))
W35 = tf.Variable(tf.zeros([784, 784]))
W36 = tf.Variable(tf.zeros([784, 784]))
W37 = tf.Variable(tf.zeros([784, 784]))
W38 = tf.Variable(tf.zeros([784, 784]))
W39 = tf.Variable(tf.zeros([784, 784]))
W40 = tf.Variable(tf.zeros([784, 784]))
W41 = tf.Variable(tf.zeros([784, 784]))
W42 = tf.Variable(tf.zeros([784, 784]))
W43 = tf.Variable(tf.zeros([784, 784]))
W44 = tf.Variable(tf.zeros([784, 784]))
W45 = tf.Variable(tf.zeros([784, 784]))
W46 = tf.Variable(tf.zeros([784, 784]))
W47 = tf.Variable(tf.zeros([784, 784]))
W48 = tf.Variable(tf.zeros([784, 784]))
W49 = tf.Variable(tf.zeros([784, 784]))
W50 = tf.Variable(tf.zeros([784, 784]))
W51 = tf.Variable(tf.zeros([784, 784]))
W52 = tf.Variable(tf.zeros([784, 784]))
W53 = tf.Variable(tf.zeros([784, 784]))
W54 = tf.Variable(tf.zeros([784, 784]))
W55 = tf.Variable(tf.zeros([784, 784]))
W56 = tf.Variable(tf.zeros([784, 784]))
W57 = tf.Variable(tf.zeros([784, 784]))
W58 = tf.Variable(tf.zeros([784, 784]))
W59 = tf.Variable(tf.zeros([784, 784]))
W60 = tf.Variable(tf.zeros([784, 784]))
W61 = tf.Variable(tf.zeros([784, 784]))
W62 = tf.Variable(tf.zeros([784, 784]))
W63 = tf.Variable(tf.zeros([784, 784]))
W64 = tf.Variable(tf.zeros([784, 784]))
W65 = tf.Variable(tf.zeros([784, 784]))
W66 = tf.Variable(tf.zeros([784, 784]))
W67 = tf.Variable(tf.zeros([784, 784]))
W68 = tf.Variable(tf.zeros([784, 784]))
W69 = tf.Variable(tf.zeros([784, 784]))
W70 = tf.Variable(tf.zeros([784, 784]))
W71 = tf.Variable(tf.zeros([784, 784]))
W72 = tf.Variable(tf.zeros([784, 784]))
W73 = tf.Variable(tf.zeros([784, 784]))
W74 = tf.Variable(tf.zeros([784, 784]))
W75 = tf.Variable(tf.zeros([784, 784]))
W76 = tf.Variable(tf.zeros([784, 784]))
W77 = tf.Variable(tf.zeros([784, 784]))
W78 = tf.Variable(tf.zeros([784, 784]))
W79 = tf.Variable(tf.zeros([784, 784]))
W80 = tf.Variable(tf.zeros([784, 784]))
W81 = tf.Variable(tf.zeros([784, 784]))
W82 = tf.Variable(tf.zeros([784, 784]))
W83 = tf.Variable(tf.zeros([784, 784]))
W84 = tf.Variable(tf.zeros([784, 784]))
W85 = tf.Variable(tf.zeros([784, 784]))
W86 = tf.Variable(tf.zeros([784, 784]))
W87 = tf.Variable(tf.zeros([784, 784]))
W88 = tf.Variable(tf.zeros([784, 784]))
W89 = tf.Variable(tf.zeros([784, 784]))
W90 = tf.Variable(tf.zeros([784, 784]))
W91 = tf.Variable(tf.zeros([784, 784]))
W92 = tf.Variable(tf.zeros([784, 784]))
W93 = tf.Variable(tf.zeros([784, 784]))
W94 = tf.Variable(tf.zeros([784, 784]))
W95 = tf.Variable(tf.zeros([784, 784]))
W96 = tf.Variable(tf.zeros([784, 784]))
W97 = tf.Variable(tf.zeros([784, 784]))
W98 = tf.Variable(tf.zeros([784, 784]))
W99 = tf.Variable(tf.zeros([784, 784]))
W100 = tf.Variable(tf.zeros([784, 784]))
W101 = tf.Variable(tf.zeros([784, 10]))

B1 = tf.Variable(tf.random_normal([784]))
B2 = tf.Variable(tf.random_normal([784]))
B3 = tf.Variable(tf.random_normal([784]))
B4 = tf.Variable(tf.random_normal([784]))
B5 = tf.Variable(tf.random_normal([784]))
B6 = tf.Variable(tf.random_normal([784]))
B7 = tf.Variable(tf.random_normal([784]))
B8 = tf.Variable(tf.random_normal([784]))
B9 = tf.Variable(tf.random_normal([784]))
B10 = tf.Variable(tf.random_normal([784]))
B11 = tf.Variable(tf.random_normal([784]))
B12 = tf.Variable(tf.random_normal([784]))
B13 = tf.Variable(tf.random_normal([784]))
B14 = tf.Variable(tf.random_normal([784]))
B15 = tf.Variable(tf.random_normal([784]))
B16 = tf.Variable(tf.random_normal([784]))
B17 = tf.Variable(tf.random_normal([784]))
B18 = tf.Variable(tf.random_normal([784]))
B19 = tf.Variable(tf.random_normal([784]))
B20 = tf.Variable(tf.random_normal([784]))
B21 = tf.Variable(tf.random_normal([784]))
B22 = tf.Variable(tf.random_normal([784]))
B23 = tf.Variable(tf.random_normal([784]))
B24 = tf.Variable(tf.random_normal([784]))
B25 = tf.Variable(tf.random_normal([784]))
B26 = tf.Variable(tf.random_normal([784]))
B27 = tf.Variable(tf.random_normal([784]))
B28 = tf.Variable(tf.random_normal([784]))
B29 = tf.Variable(tf.random_normal([784]))
B30 = tf.Variable(tf.random_normal([784]))
B31 = tf.Variable(tf.random_normal([784]))
B32 = tf.Variable(tf.random_normal([784]))
B33 = tf.Variable(tf.random_normal([784]))
B34 = tf.Variable(tf.random_normal([784]))
B35 = tf.Variable(tf.random_normal([784]))
B36 = tf.Variable(tf.random_normal([784]))
B37 = tf.Variable(tf.random_normal([784]))
B38 = tf.Variable(tf.random_normal([784]))
B39 = tf.Variable(tf.random_normal([784]))
B40 = tf.Variable(tf.random_normal([784]))
B41 = tf.Variable(tf.random_normal([784]))
B42 = tf.Variable(tf.random_normal([784]))
B43 = tf.Variable(tf.random_normal([784]))
B44 = tf.Variable(tf.random_normal([784]))
B45 = tf.Variable(tf.random_normal([784]))
B46 = tf.Variable(tf.random_normal([784]))
B47 = tf.Variable(tf.random_normal([784]))
B48 = tf.Variable(tf.random_normal([784]))
B49 = tf.Variable(tf.random_normal([784]))
B50 = tf.Variable(tf.random_normal([784]))
B51 = tf.Variable(tf.random_normal([784]))
B52 = tf.Variable(tf.random_normal([784]))
B53 = tf.Variable(tf.random_normal([784]))
B54 = tf.Variable(tf.random_normal([784]))
B55 = tf.Variable(tf.random_normal([784]))
B56 = tf.Variable(tf.random_normal([784]))
B57 = tf.Variable(tf.random_normal([784]))
B58 = tf.Variable(tf.random_normal([784]))
B59 = tf.Variable(tf.random_normal([784]))
B60 = tf.Variable(tf.random_normal([784]))
B61 = tf.Variable(tf.random_normal([784]))
B62 = tf.Variable(tf.random_normal([784]))
B63 = tf.Variable(tf.random_normal([784]))
B64 = tf.Variable(tf.random_normal([784]))
B65 = tf.Variable(tf.random_normal([784]))
B66 = tf.Variable(tf.random_normal([784]))
B67 = tf.Variable(tf.random_normal([784]))
B68 = tf.Variable(tf.random_normal([784]))
B69 = tf.Variable(tf.random_normal([784]))
B70 = tf.Variable(tf.random_normal([784]))
B71 = tf.Variable(tf.random_normal([784]))
B72 = tf.Variable(tf.random_normal([784]))
B73 = tf.Variable(tf.random_normal([784]))
B74 = tf.Variable(tf.random_normal([784]))
B75 = tf.Variable(tf.random_normal([784]))
B76 = tf.Variable(tf.random_normal([784]))
B77 = tf.Variable(tf.random_normal([784]))
B78 = tf.Variable(tf.random_normal([784]))
B79 = tf.Variable(tf.random_normal([784]))
B80 = tf.Variable(tf.random_normal([784]))
B81 = tf.Variable(tf.random_normal([784]))
B82 = tf.Variable(tf.random_normal([784]))
B83 = tf.Variable(tf.random_normal([784]))
B84 = tf.Variable(tf.random_normal([784]))
B85 = tf.Variable(tf.random_normal([784]))
B86 = tf.Variable(tf.random_normal([784]))
B87 = tf.Variable(tf.random_normal([784]))
B88 = tf.Variable(tf.random_normal([784]))
B89 = tf.Variable(tf.random_normal([784]))
B90 = tf.Variable(tf.random_normal([784]))
B91 = tf.Variable(tf.random_normal([784]))
B92 = tf.Variable(tf.random_normal([784]))
B93 = tf.Variable(tf.random_normal([784]))
B94 = tf.Variable(tf.random_normal([784]))
B95 = tf.Variable(tf.random_normal([784]))
B96 = tf.Variable(tf.random_normal([784]))
B97 = tf.Variable(tf.random_normal([784]))
B98 = tf.Variable(tf.random_normal([784]))
B99 = tf.Variable(tf.random_normal([784]))
B100 = tf.Variable(tf.random_normal([784]))
B101 = tf.Variable(tf.random_normal([10]))

L1 = tf.nn.relu(tf.add(tf.matmul(L0_sc, W1), B1))
L1_sc = tf.add(L0_sc, L1)
L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), B2))
L2_sc = tf.add(L1_sc, L2)
L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), B3))
L3_sc = tf.add(L2_sc, L3)
L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), B4))
L4_sc = tf.add(L3_sc, L4)
L5 = tf.nn.relu(tf.add(tf.matmul(L4, W5), B5))
L5_sc = tf.add(L4_sc, L5)
L6 = tf.nn.relu(tf.add(tf.matmul(L5, W6), B6))
L6_sc = tf.add(L5_sc, L6)
L7 = tf.nn.relu(tf.add(tf.matmul(L6, W7), B7))
L7_sc = tf.add(L6_sc, L7)
L8 = tf.nn.relu(tf.add(tf.matmul(L7, W8), B8))
L8_sc = tf.add(L7_sc, L8)
L9 = tf.nn.relu(tf.add(tf.matmul(L8, W9), B9))
L9_sc = tf.add(L8_sc, L9)
L10 = tf.nn.relu(tf.add(tf.matmul(L9, W10), B10))
L10_sc = tf.add(L9_sc, L10)
L11 = tf.nn.relu(tf.add(tf.matmul(L10, W11), B11))
L11_sc = tf.add(L10_sc, L11)
L12 = tf.nn.relu(tf.add(tf.matmul(L11, W12), B12))
L12_sc = tf.add(L11_sc, L12)
L13 = tf.nn.relu(tf.add(tf.matmul(L12, W13), B13))
L13_sc = tf.add(L12_sc, L13)
L14 = tf.nn.relu(tf.add(tf.matmul(L13, W14), B14))
L14_sc = tf.add(L13_sc, L14)
L15 = tf.nn.relu(tf.add(tf.matmul(L14, W15), B15))
L15_sc = tf.add(L14_sc, L15)
L16 = tf.nn.relu(tf.add(tf.matmul(L15, W16), B16))
L16_sc = tf.add(L15_sc, L16)
L17 = tf.nn.relu(tf.add(tf.matmul(L16, W17), B17))
L17_sc = tf.add(L16_sc, L17)
L18 = tf.nn.relu(tf.add(tf.matmul(L17, W18), B18))
L18_sc = tf.add(L17_sc, L18)
L19 = tf.nn.relu(tf.add(tf.matmul(L18, W19), B19))
L19_sc = tf.add(L18_sc, L19)
L20 = tf.nn.relu(tf.add(tf.matmul(L19, W20), B20))
L20_sc = tf.add(L19_sc, L20)
L21 = tf.nn.relu(tf.add(tf.matmul(L20, W21), B21))
L21_sc = tf.add(L20_sc, L21)
L22 = tf.nn.relu(tf.add(tf.matmul(L21, W22), B22))
L22_sc = tf.add(L21_sc, L22)
L23 = tf.nn.relu(tf.add(tf.matmul(L22, W23), B23))
L23_sc = tf.add(L22_sc, L23)
L24 = tf.nn.relu(tf.add(tf.matmul(L23, W24), B24))
L24_sc = tf.add(L23_sc, L24)
L25 = tf.nn.relu(tf.add(tf.matmul(L24, W25), B25))
L25_sc = tf.add(L24_sc, L25)
L26 = tf.nn.relu(tf.add(tf.matmul(L25, W26), B26))
L26_sc = tf.add(L25_sc, L26)
L27 = tf.nn.relu(tf.add(tf.matmul(L26, W27), B27))
L27_sc = tf.add(L26_sc, L27)
L28 = tf.nn.relu(tf.add(tf.matmul(L27, W28), B28))
L28_sc = tf.add(L27_sc, L28)
L29 = tf.nn.relu(tf.add(tf.matmul(L28, W29), B29))
L29_sc = tf.add(L28_sc, L29)
L30 = tf.nn.relu(tf.add(tf.matmul(L29, W30), B30))
L30_sc = tf.add(L29_sc, L30)
L31 = tf.nn.relu(tf.add(tf.matmul(L30, W31), B31))
L31_sc = tf.add(L30_sc, L31)
L32 = tf.nn.relu(tf.add(tf.matmul(L31, W32), B32))
L32_sc = tf.add(L31_sc, L32)
L33 = tf.nn.relu(tf.add(tf.matmul(L32, W33), B33))
L33_sc = tf.add(L32_sc, L33)
L34 = tf.nn.relu(tf.add(tf.matmul(L33, W34), B34))
L34_sc = tf.add(L33_sc, L34)
L35 = tf.nn.relu(tf.add(tf.matmul(L34, W35), B35))
L35_sc = tf.add(L34_sc, L35)
L36 = tf.nn.relu(tf.add(tf.matmul(L35, W36), B36))
L36_sc = tf.add(L35_sc, L36)
L37 = tf.nn.relu(tf.add(tf.matmul(L36, W37), B37))
L37_sc = tf.add(L36_sc, L37)
L38 = tf.nn.relu(tf.add(tf.matmul(L37, W38), B38))
L38_sc = tf.add(L37_sc, L38)
L39 = tf.nn.relu(tf.add(tf.matmul(L38, W39), B39))
L39_sc = tf.add(L38_sc, L39)
L40 = tf.nn.relu(tf.add(tf.matmul(L39, W40), B40))
L40_sc = tf.add(L39_sc, L40)
L41 = tf.nn.relu(tf.add(tf.matmul(L40, W41), B41))
L41_sc = tf.add(L40_sc, L41)
L42 = tf.nn.relu(tf.add(tf.matmul(L41, W42), B42))
L42_sc = tf.add(L41_sc, L42)
L43 = tf.nn.relu(tf.add(tf.matmul(L42, W43), B43))
L43_sc = tf.add(L42_sc, L43)
L44 = tf.nn.relu(tf.add(tf.matmul(L43, W44), B44))
L44_sc = tf.add(L43_sc, L44)
L45 = tf.nn.relu(tf.add(tf.matmul(L44, W45), B45))
L45_sc = tf.add(L44_sc, L45)
L46 = tf.nn.relu(tf.add(tf.matmul(L45, W46), B46))
L46_sc = tf.add(L45_sc, L46)
L47 = tf.nn.relu(tf.add(tf.matmul(L46, W47), B47))
L47_sc = tf.add(L46_sc, L47)
L48 = tf.nn.relu(tf.add(tf.matmul(L47, W48), B48))
L48_sc = tf.add(L47_sc, L48)
L49 = tf.nn.relu(tf.add(tf.matmul(L48, W49), B49))
L49_sc = tf.add(L48_sc, L49)
L50 = tf.nn.relu(tf.add(tf.matmul(L49, W50), B50))
L50_sc = tf.add(L49_sc, L50)
L51 = tf.nn.relu(tf.add(tf.matmul(L50, W51), B51))
L51_sc = tf.add(L50_sc, L51)
L52 = tf.nn.relu(tf.add(tf.matmul(L51, W52), B52))
L52_sc = tf.add(L51_sc, L52)
L53 = tf.nn.relu(tf.add(tf.matmul(L52, W53), B53))
L53_sc = tf.add(L52_sc, L53)
L54 = tf.nn.relu(tf.add(tf.matmul(L53, W54), B54))
L54_sc = tf.add(L53_sc, L54)
L55 = tf.nn.relu(tf.add(tf.matmul(L54, W55), B55))
L55_sc = tf.add(L54_sc, L55)
L56 = tf.nn.relu(tf.add(tf.matmul(L55, W56), B56))
L56_sc = tf.add(L55_sc, L56)
L57 = tf.nn.relu(tf.add(tf.matmul(L56, W57), B57))
L57_sc = tf.add(L56_sc, L57)
L58 = tf.nn.relu(tf.add(tf.matmul(L57, W58), B58))
L58_sc = tf.add(L57_sc, L58)
L59 = tf.nn.relu(tf.add(tf.matmul(L58, W59), B59))
L59_sc = tf.add(L58_sc, L59)
L60 = tf.nn.relu(tf.add(tf.matmul(L59, W60), B60))
L60_sc = tf.add(L59_sc, L60)
L61 = tf.nn.relu(tf.add(tf.matmul(L60, W61), B61))
L61_sc = tf.add(L60_sc, L61)
L62 = tf.nn.relu(tf.add(tf.matmul(L61, W62), B62))
L62_sc = tf.add(L61_sc, L62)
L63 = tf.nn.relu(tf.add(tf.matmul(L62, W63), B63))
L63_sc = tf.add(L62_sc, L63)
L64 = tf.nn.relu(tf.add(tf.matmul(L63, W64), B64))
L64_sc = tf.add(L63_sc, L64)
L65 = tf.nn.relu(tf.add(tf.matmul(L64, W65), B65))
L65_sc = tf.add(L64_sc, L65)
L66 = tf.nn.relu(tf.add(tf.matmul(L65, W66), B66))
L66_sc = tf.add(L65_sc, L66)
L67 = tf.nn.relu(tf.add(tf.matmul(L66, W67), B67))
L67_sc = tf.add(L66_sc, L67)
L68 = tf.nn.relu(tf.add(tf.matmul(L67, W68), B68))
L68_sc = tf.add(L67_sc, L68)
L69 = tf.nn.relu(tf.add(tf.matmul(L68, W69), B69))
L69_sc = tf.add(L68_sc, L69)
L70 = tf.nn.relu(tf.add(tf.matmul(L69, W70), B70))
L70_sc = tf.add(L69_sc, L70)
L71 = tf.nn.relu(tf.add(tf.matmul(L70, W71), B71))
L71_sc = tf.add(L70_sc, L71)
L72 = tf.nn.relu(tf.add(tf.matmul(L71, W72), B72))
L72_sc = tf.add(L71_sc, L72)
L73 = tf.nn.relu(tf.add(tf.matmul(L72, W73), B73))
L73_sc = tf.add(L72_sc, L73)
L74 = tf.nn.relu(tf.add(tf.matmul(L73, W74), B74))
L74_sc = tf.add(L73_sc, L74)
L75 = tf.nn.relu(tf.add(tf.matmul(L74, W75), B75))
L75_sc = tf.add(L74_sc, L75)
L76 = tf.nn.relu(tf.add(tf.matmul(L75, W76), B76))
L76_sc = tf.add(L75_sc, L76)
L77 = tf.nn.relu(tf.add(tf.matmul(L76, W77), B77))
L77_sc = tf.add(L76_sc, L77)
L78 = tf.nn.relu(tf.add(tf.matmul(L77, W78), B78))
L78_sc = tf.add(L77_sc, L78)
L79 = tf.nn.relu(tf.add(tf.matmul(L78, W79), B79))
L79_sc = tf.add(L78_sc, L79)
L80 = tf.nn.relu(tf.add(tf.matmul(L79, W80), B80))
L80_sc = tf.add(L79_sc, L80)
L81 = tf.nn.relu(tf.add(tf.matmul(L80, W81), B81))
L81_sc = tf.add(L80_sc, L81)
L82 = tf.nn.relu(tf.add(tf.matmul(L81, W82), B82))
L82_sc = tf.add(L81_sc, L82)
L83 = tf.nn.relu(tf.add(tf.matmul(L82, W83), B83))
L83_sc = tf.add(L82_sc, L83)
L84 = tf.nn.relu(tf.add(tf.matmul(L83, W84), B84))
L84_sc = tf.add(L83_sc, L84)
L85 = tf.nn.relu(tf.add(tf.matmul(L84, W85), B85))
L85_sc = tf.add(L84_sc, L85)
L86 = tf.nn.relu(tf.add(tf.matmul(L85, W86), B86))
L86_sc = tf.add(L85_sc, L86)
L87 = tf.nn.relu(tf.add(tf.matmul(L86, W87), B87))
L87_sc = tf.add(L86_sc, L87)
L88 = tf.nn.relu(tf.add(tf.matmul(L87, W88), B88))
L88_sc = tf.add(L87_sc, L88)
L89 = tf.nn.relu(tf.add(tf.matmul(L88, W89), B89))
L89_sc = tf.add(L88_sc, L89)
L90 = tf.nn.relu(tf.add(tf.matmul(L89, W90), B90))
L90_sc = tf.add(L89_sc, L90)
L91 = tf.nn.relu(tf.add(tf.matmul(L90, W91), B91))
L91_sc = tf.add(L90_sc, L91)
L92 = tf.nn.relu(tf.add(tf.matmul(L91, W92), B92))
L92_sc = tf.add(L91_sc, L92)
L93 = tf.nn.relu(tf.add(tf.matmul(L92, W93), B93))
L93_sc = tf.add(L92_sc, L93)
L94 = tf.nn.relu(tf.add(tf.matmul(L93, W94), B94))
L94_sc = tf.add(L93_sc, L94)
L95 = tf.nn.relu(tf.add(tf.matmul(L94, W95), B95))
L95_sc = tf.add(L94_sc, L95)
L96 = tf.nn.relu(tf.add(tf.matmul(L95, W96), B96))
L96_sc = tf.add(L95_sc, L96)
L97 = tf.nn.relu(tf.add(tf.matmul(L96, W97), B97))
L97_sc = tf.add(L96_sc, L97)
L98 = tf.nn.relu(tf.add(tf.matmul(L97, W98), B98))
L98_sc = tf.add(L97_sc, L98)
L99 = tf.nn.relu(tf.add(tf.matmul(L98, W99), B99))
L99_sc = tf.add(L98_sc, L99)
L100 = tf.nn.relu(tf.add(tf.matmul(L99, W100), B100))
L100_sc = tf.add(L99_sc, L100)
hypothesis = tf.add(tf.matmul(L100_sc, W101), B101)

val = tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = hypothesis)
cost = tf.reduce_mean(val)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for step in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict = {L0_sc: batch_xs, Y:batch_ys})

            avg_cost += sess.run(cost, feed_dict = {L0_sc:batch_xs, Y:batch_ys})/total_batch

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print("Accuracy=", accuracy.eval({L0_sc:mnist.test.images, Y:mnist.test.labels}), "normal_dnn")