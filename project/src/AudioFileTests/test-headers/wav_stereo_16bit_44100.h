#include <vector>

namespace wav_stereo_16bit_44100 { 

int numSamplesPerChannel = 352800;
int bitDepth = 16;
uint32_t sampleRate = 44100;
int numChannels = 2;

std::vector<std::vector<double>> testBuffer = {{0.0, -9.1552734375e-05, -0.00054931640625, -0.00119018554688, -0.00128173828125, 0.00198364257812, 0.0105590820312, 0.0144958496094, 0.009765625, -0.004638671875, -0.0225219726562, -0.0165100097656, 0.00762939453125, 0.0166931152344, 0.00927734375, 0.00332641601562, 0.00128173828125, -0.00262451171875, -0.00296020507812, -0.0047607421875, -0.00442504882812, -0.0209045410156, -0.0406494140625, -0.0335693359375, -0.029541015625, -0.0284729003906, -0.0218200683594, -0.0181579589844, -0.0096435546875, 0.0116882324219, 0.0351867675781, 0.0457763671875, 0.045166015625, 0.0595703125, 0.0734558105469, 0.0656127929688, 0.0718688964844, 0.059326171875, 0.0093994140625, -0.032958984375, 0.0137329101562, -0.0510559082031, 0.0336608886719, 0.107086181641, -0.042236328125, 0.0374755859375, -0.0726928710938, 0.0368347167969, 0.106475830078, -0.0514831542969, 0.0101623535156, -0.0618286132812, 0.0856628417969, 0.104461669922, 0.0771789550781, 0.0501098632812, 0.0333251953125, 0.138336181641, 0.124298095703, 0.172241210938, 0.084716796875, 0.103210449219, 0.0732421875, 0.0639343261719, 0.107696533203, 0.0631408691406, 0.0463562011719, -0.0134582519531, 0.04345703125, 0.0827026367188, 0.109619140625, 0.0625610351562, 0.0874633789062, 0.105041503906, 0.0605773925781, 0.0354309082031, -0.0451965332031, -0.0210571289062, -0.0549011230469, 0.00106811523438, 0.0806274414062, 0.0448303222656, 0.015380859375, 0.0315856933594, 0.108459472656, 0.187469482422, 0.161468505859, 0.0662536621094, 0.0614318847656, 0.0788269042969, 0.0732116699219, -0.0542907714844, -0.0577087402344, -0.00146484375, -0.0479431152344, 0.0653076171875, 0.0799255371094, 0.00119018554688, 0.0883178710938, 0.0580444335938, 0.0507202148438, 0.14404296875, 0.0220031738281, 0.0840148925781, 0.122406005859, 0.122192382812, 0.154815673828, 0.00149536132812, 0.0361328125, 0.0135192871094, 0.0263671875, 0.02099609375, -0.0580139160156, -0.041015625, -0.0712890625, -0.0126953125, 0.00289916992188, 0.00677490234375, -0.008544921875, -0.0557861328125, -0.0612182617188, -0.0353698730469, -0.00250244140625, 0.034912109375, 0.065673828125, 0.00137329101562, -0.0312194824219, -0.120910644531, -0.159271240234, -0.15771484375, -0.233642578125, -0.179382324219, -0.133697509766, -0.107849121094, -0.122406005859, -0.138427734375, -0.0659790039062, -0.088623046875, -0.0758666992188, -0.0291748046875, -0.0794982910156, -0.0418090820312, -0.0246276855469, -0.0798950195312, -0.000457763671875, 0.0233154296875, 0.0270385742188, 0.0595092773438, -0.001220703125, 0.000335693359375, -0.00848388671875, -0.030029296875, -0.0479736328125, -0.0628051757812, -0.0783996582031, -0.0804748535156, -0.00143432617188, -0.0500183105469, -0.0865173339844, -0.074462890625, -0.0792846679688, -0.04638671875, -0.0882263183594, -0.0945129394531, -0.0933532714844, -0.112884521484, -0.0833740234375, -0.0848083496094, -0.0939025878906, -0.0563049316406, -0.0721435546875, -0.095947265625, -0.0603942871094, -0.0759887695312, -0.0631408691406, -0.0370483398438, -0.0552368164062, -0.052490234375, -0.0558776855469, -0.0589294433594, -0.0558776855469, -0.0559997558594, -0.0638732910156, -0.0585632324219, -0.0386352539062, -0.0389404296875, 0.0113220214844, 0.0577087402344, 0.0721130371094, 0.154846191406, 0.141021728516, 0.0987854003906, 0.16357421875, 0.131408691406, 0.0897216796875, 0.0821228027344, 0.0105590820312, -0.00942993164062, -0.0130920410156, -0.0369262695312, -0.05224609375, -0.0833740234375, -0.0537719726562, -0.0376892089844, -0.0652160644531, -0.0292358398438, -0.0408630371094, -0.0507202148438, 0.0263366699219, 0.0471496582031, 0.0354614257812, 0.0350036621094, 0.0607299804688, 0.0708923339844, 0.039794921875, 0.064453125, 0.0325622558594, 0.0161437988281, 0.048583984375, 0.0120849609375, 0.0284423828125, 0.0232238769531, -0.0136413574219, 0.00332641601562, -0.00607299804688, -0.0259399414062, -0.0177917480469, 0.00363159179688, 0.000701904296875, -0.0308227539062, -0.0197448730469, -0.0122375488281, -0.0218200683594, 0.0211791992188, 0.0371398925781, -0.00149536132812, 0.0194702148438, 0.0337524414062, 0.0155029296875, 0.0637512207031, 0.0529479980469, 0.0922546386719, 0.120910644531, 0.0977172851562, 0.142486572266, 0.0881042480469, 0.109802246094, 0.122863769531, 0.0924987792969, 0.0785217285156, 0.0424194335938, 0.0885925292969, 0.101135253906, 0.0919494628906, 0.093505859375, 0.0860290527344, 0.0690002441406, 0.0987854003906, 0.0873107910156, 0.0827941894531, 0.0638732910156, 0.038330078125, 0.0472106933594, 0.013427734375, 0.04443359375, 0.007080078125, 0.00997924804688, -0.0164184570312, -0.010986328125, -0.0123901367188, -0.0314636230469, 0.0238342285156, -0.0289306640625, 0.00579833984375, -0.0467224121094, -0.0147705078125, 0.0181579589844, 0.0215148925781, 0.0285034179688, -0.00686645507812, 0.0512390136719, -0.05078125, -0.0111083984375, -0.00936889648438, 0.0131225585938, 0.0192260742188, -0.013427734375, 0.0438232421875, -0.0783996582031, -0.0490417480469, -0.0335388183594, -0.0458679199219, -0.03564453125, -0.0486450195312, -0.0646667480469, -0.068359375, -0.0717468261719, -0.0931396484375, -0.0380859375, -0.0853881835938, -0.0657043457031, -0.0820617675781, -0.0695495605469, -0.054931640625, -0.100708007812, -0.0575866699219, -0.114624023438, -0.09619140625, -0.125915527344, -0.116729736328, -0.0756530761719, -0.114990234375, -0.05810546875, -0.1025390625, -0.0735778808594, -0.0856018066406, -0.138000488281, -0.0701904296875, -0.148498535156, -0.151428222656, -0.131988525391, -0.113830566406, -0.110534667969, -0.161376953125, -0.110290527344, -0.135498046875, -0.116943359375, -0.084228515625, -0.0194396972656, -0.0371704101562, -0.0447387695312, 0.0430908203125, -0.0109252929688, 0.0663757324219, 0.0555114746094, 0.0622253417969, 0.0639038085938, -0.0230102539062, 0.0374450683594, 0.0249938964844, 0.0657348632812, 0.0367431640625, 0.00637817382812, 0.0453186035156, -0.0339050292969, 0.0159606933594, 0.0101623535156, -0.0350646972656, 0.0398864746094, -0.0411987304688, -0.00555419921875, 0.0078125, -0.0499877929688, 0.00558471679688, -0.042724609375, 0.0107421875, -0.00161743164062, 0.00384521484375, 0.0373229980469, -0.0127868652344, 0.0126037597656, -0.0224304199219, -0.0170593261719, -0.0213928222656, -0.0479431152344, -0.0219116210938, 0.0144958496094, -0.00399780273438, -0.0148620605469, -0.0169067382812, -0.00888061523438, 0.00534057617188, -0.0242309570312, 0.0180969238281, 0.00567626953125, -0.0106201171875, -0.00958251953125, -0.016845703125, 0.00662231445312, 0.000335693359375, 0.0013427734375, 0.0680847167969, 0.0467224121094, 0.05419921875, 0.147399902344, 0.0347900390625, 0.083740234375, 0.0494384765625, 0.0187377929688, 0.119903564453, -0.0262145996094, 0.0846252441406, 0.116516113281, 0.0402526855469, 0.07470703125, 0.0350341796875, 0.0805358886719, 0.0827026367188, 0.0537414550781, 0.0714416503906, 0.127502441406, 0.133880615234, 0.138366699219, 0.129852294922, 0.0904541015625, 0.086181640625, 0.0922241210938, 0.074462890625, 0.0079345703125, 0.0362243652344, -0.00897216796875, -0.0154113769531, 0.0174255371094, -0.0530700683594, -0.0127258300781, -0.00875854492188, 0.00949096679688, 0.0252075195312, 0.035400390625, 0.0613098144531, 0.0281066894531, 0.0649719238281, 0.0917358398438, 0.0642395019531, 0.0374450683594, 0.071044921875, -0.00439453125, -0.0290832519531, 0.0129089355469, -0.0229797363281, 0.025634765625, -0.0479431152344, -0.0173645019531, 0.0161743164062, -0.0498046875, -0.0222473144531, -0.0245971679688, 0.0163269042969, -0.0391845703125, -0.0567016601562, -0.0283508300781, -0.0292663574219, -0.0130004882812, -0.0625915527344, -0.0620422363281, -0.0543518066406, -0.0494689941406, -0.0545349121094, -0.0547790527344, -0.0287780761719, -0.0343627929688, -0.0391845703125, -0.0374450683594, -0.0628356933594, -0.08154296875, -0.06982421875, -0.092529296875, -0.117095947266, -0.108154296875, -0.120971679688, -0.140319824219, -0.110229492188, -0.109161376953, -0.0985717773438, -0.0581665039062, -0.0671691894531, -0.0689086914062, -0.0635375976562, -0.0526123046875, -0.0362243652344, -0.0203857421875, -0.0113220214844, -0.0211791992188, -0.0390014648438, -0.0441284179688, -0.0328063964844, -0.0436096191406, -0.0337219238281, -0.0181579589844, -0.041748046875, -0.017578125, 0.0009765625, -0.018310546875, -0.0189208984375, -0.0645141601562, -0.0726623535156, -0.0556945800781, -0.0848388671875, -0.0726928710938, -0.0704956054688, -0.0993347167969, -0.0720825195312, -0.0634460449219, -0.0672912597656, -0.0127258300781, -0.0311584472656, -0.0663146972656, -0.0717163085938, -0.0643615722656, -0.0353698730469}, {0.0, -9.1552734375e-05, -0.000640869140625, -0.00152587890625, -0.00173950195312, 0.0018310546875, 0.01220703125, 0.018310546875, 0.01416015625, -0.00228881835938, -0.0221252441406, -0.0171203613281, 0.00592041015625, 0.0142211914062, 0.00820922851562, 0.00497436523438, 0.00332641601562, -0.00289916992188, -0.00595092773438, -0.00973510742188, -0.00970458984375, -0.0254821777344, -0.0470275878906, -0.0365295410156, -0.0367126464844, -0.010009765625, 0.100830078125, 0.0028076171875, 0.0911865234375, 0.0331420898438, 0.103912353516, 0.0850219726562, 0.0938720703125, 0.110626220703, 0.103912353516, 0.135375976562, 0.0794372558594, 0.11279296875, 0.0551147460938, 0.0516052246094, -0.0141906738281, 0.0108337402344, 0.0167236328125, 0.026123046875, 0.0210571289062, -0.0235900878906, 0.0115661621094, -0.000152587890625, 0.0427856445312, -0.0264587402344, -0.0403747558594, -0.0748291015625, -0.00210571289062, 0.0427551269531, -0.0018310546875, -0.00643920898438, 0.0162048339844, 0.121276855469, 0.115051269531, 0.157501220703, 0.156585693359, 0.158447265625, 0.0720520019531, 0.0967102050781, 0.124755859375, 0.0299377441406, 0.0707702636719, 0.0315246582031, 0.0401611328125, 0.125366210938, 0.127136230469, 0.0857543945312, 0.109741210938, 0.0620727539062, 0.0907592773438, -0.0177001953125, -0.0505065917969, 0.0285949707031, -0.0672302246094, 0.0105590820312, 0.0538024902344, 0.0663146972656, 0.00442504882812, -0.0123291015625, 0.0321350097656, 0.128356933594, 0.102233886719, 0.04541015625, 0.0635986328125, 0.00497436523438, 0.00341796875, -0.103698730469, -0.105895996094, -0.112640380859, -0.0395812988281, -0.0310668945312, 0.0206909179688, 0.0426940917969, 0.0430908203125, 0.1123046875, 0.0504760742188, 0.141387939453, 0.0513610839844, 0.0870666503906, 0.0469360351562, 0.0795593261719, 0.0545654296875, -0.0701904296875, -0.0380249023438, -0.0620727539062, -0.0287780761719, -0.134460449219, -0.0302429199219, -0.1181640625, -0.0427856445312, -0.031982421875, -0.0799255371094, 0.0322570800781, -0.0775756835938, -0.0716552734375, -0.0707397460938, -0.0364074707031, -0.0604248046875, 0.0372619628906, -0.0224914550781, 0.00628662109375, -0.0813293457031, -0.11474609375, -0.087158203125, -0.182189941406, -0.0345764160156, -0.140441894531, 0.00833129882812, -0.0584411621094, -0.0469055175781, -0.0271301269531, -0.0998229980469, -0.0195007324219, -0.121948242188, -0.0256652832031, -0.107635498047, -0.0196533203125, -0.0529479980469, -0.0609436035156, -0.0169067382812, -0.0634460449219, 0.0367736816406, 0.00753784179688, 0.0661010742188, 0.0321655273438, 0.0412902832031, 0.00936889648438, -0.00564575195312, -0.0350646972656, -0.0838623046875, -0.0692138671875, -0.139434814453, -0.0965576171875, -0.0858764648438, -0.126922607422, -0.074462890625, -0.0316162109375, -0.0569763183594, -0.0205993652344, -0.0166625976562, 0.0256042480469, 0.00393676757812, -0.0148315429688, 0.0493774414062, 0.00399780273438, 0.0217895507812, 0.00836181640625, -0.0584106445312, -0.03564453125, -0.0278930664062, -0.0378112792969, -0.00299072265625, -0.0750427246094, -0.0341491699219, 0.00665283203125, 0.00521850585938, -0.010986328125, -0.0513610839844, 0.0182189941406, -0.0157775878906, 0.0447387695312, 0.0332946777344, 0.0532531738281, 0.0918884277344, 0.0950317382812, 0.0922546386719, 0.0272827148438, 0.0912475585938, 0.0724182128906, 0.0949401855469, 0.0600280761719, 0.0427856445312, 0.0352478027344, 0.0243530273438, 0.0270690917969, 0.0113220214844, 0.0442199707031, 0.0127258300781, 0.048828125, 0.0224914550781, 0.0476379394531, 0.0281066894531, 0.0183715820312, 0.0611267089844, 0.0422058105469, 0.0431213378906, 0.0603637695312, 0.0647583007812, 0.00772094726562, 0.0696411132812, 0.0489807128906, 0.0677795410156, 0.0823364257812, 0.0846862792969, 0.11572265625, 0.0615234375, 0.0677795410156, 0.00588989257812, -0.00247192382812, -0.0210876464844, -0.0548400878906, -0.0644226074219, -0.043212890625, 0.0105285644531, -0.00808715820312, -0.0498352050781, -0.0234069824219, 0.0233764648438, 0.0183410644531, 0.0480041503906, 0.0247802734375, 0.0157470703125, 0.0213928222656, 0.0487976074219, 0.0435485839844, 0.0142517089844, 0.0287475585938, 0.00341796875, 0.0264587402344, 0.0264892578125, 0.0370788574219, 0.0360107421875, 0.0244750976562, 0.0182189941406, 0.00784301757812, 0.0111694335938, 0.00225830078125, 0.0277099609375, 0.00241088867188, -0.0223083496094, -0.0125732421875, -0.0326232910156, -0.0674438476562, -0.0618896484375, -0.0532836914062, -0.0729064941406, -0.0713806152344, -0.0760192871094, -0.0939025878906, -0.069091796875, -0.0569458007812, -0.0790100097656, -0.0487976074219, -0.0438842773438, -0.0729370117188, -0.070068359375, -0.055908203125, -0.0397338867188, -0.0462341308594, -0.0652160644531, -0.0587768554688, -0.0470886230469, -0.02880859375, -0.0331420898438, -0.0303955078125, -0.0255126953125, -0.025390625, -0.019287109375, -0.0303039550781, -0.00729370117188, -0.0256042480469, -0.0305786132812, -0.0408630371094, -0.0419921875, -0.0418090820312, -0.0413818359375, -0.0181579589844, -0.0689392089844, -0.08447265625, -0.0984497070312, -0.0679626464844, -0.0796813964844, -0.115417480469, -0.0947875976562, -0.05859375, -0.0870971679688, -0.0857543945312, -0.0143432617188, -0.0758056640625, -0.0628662109375, -0.0676879882812, -0.0745544433594, -0.0948181152344, -0.0494384765625, 0.00875854492188, -0.0315856933594, -0.0126342773438, -0.0281677246094, -0.0132141113281, -0.0562133789062, -0.0202026367188, -0.0224304199219, -0.0106811523438, -0.03125, -0.0656127929688, -0.00787353515625, -0.0288391113281, 0.00799560546875, -0.0430908203125, 0.00430297851562, -0.0180969238281, 0.0106201171875, 0.0221862792969, -0.00128173828125, 0.0835876464844, 0.0543212890625, 0.110626220703, 0.0461730957031, 0.0702209472656, 0.0613708496094, 0.0661315917969, 0.0906372070312, 0.0397644042969, 0.0591735839844, 0.0327758789062, 0.0885620117188, 0.0785522460938, 0.0728454589844, 0.0806884765625, 0.0940856933594, 0.0354614257812, 0.0586547851562, 0.0824584960938, 0.02001953125, 0.0224914550781, -0.0165100097656, 0.0186462402344, 0.0296630859375, 0.0359802246094, 0.0409851074219, 0.0677490234375, 0.0147094726562, 0.0734252929688, 0.094482421875, 0.0217895507812, 0.0519409179688, 0.00131225585938, 0.0589599609375, 0.0251770019531, 0.038818359375, 0.0581359863281, 0.0474853515625, 0.0751647949219, 0.0250854492188, 0.0437622070312, 0.0511779785156, 0.0138244628906, 0.001708984375, 0.0278015136719, 0.0258178710938, 0.0440063476562, 0.0239562988281, 0.0298767089844, 0.00372314453125, 0.00436401367188, -0.0169372558594, -0.00579833984375, 0.0392761230469, 0.037841796875, 0.0544128417969, -0.02734375, 0.00213623046875, -0.0336303710938, 0.00161743164062, -0.0135803222656, 0.0067138671875, 0.0126953125, -0.0167846679688, 0.0170288085938, -0.0739440917969, 0.00616455078125, -0.0510559082031, 0.0436096191406, -0.0029296875, 0.008056640625, 0.0367431640625, -0.0478515625, 0.042236328125, 0.00408935546875, 0.0399780273438, -0.0648498535156, 0.0422058105469, -0.0282592773438, -0.0122985839844, -0.0171813964844, -0.0295715332031, 0.05615234375, -0.110900878906, -0.0173645019531, -0.079833984375, -0.00347900390625, -0.0732116699219, -0.0600280761719, -0.00228881835938, -0.0486450195312, -0.00637817382812, -0.100372314453, -0.0252685546875, -0.0593872070312, -0.00973510742188, -0.0568542480469, -0.0691833496094, -0.0697326660156, -0.0488891601562, -0.0602111816406, -0.1201171875, -0.0810852050781, -0.142456054688, -0.0755920410156, -0.135559082031, -0.102020263672, -0.173645019531, -0.141662597656, -0.0902099609375, -0.164672851562, -0.0675964355469, -0.13037109375, -0.0834655761719, -0.0703735351562, -0.0571899414062, -0.0343627929688, -0.03662109375, -0.0535888671875, -0.0323486328125, -0.0101623535156, -0.0741882324219, -0.0296630859375, -0.0392150878906, -0.0382080078125, -0.0345153808594, -0.0278930664062, -0.0045166015625, -0.0484619140625, -0.0625, -0.00299072265625, -0.0388488769531, -0.0220947265625, -0.0201416015625, -0.0280456542969, 0.0057373046875, -0.0605163574219, 0.01171875, 0.00784301757812, 0.00872802734375, -0.0140686035156, -0.0230712890625, 0.00115966796875, -0.0300903320312, 0.0144653320312, -0.012451171875, 0.0135498046875, 0.0247497558594, 0.0346984863281, 0.019287109375, 0.0311889648438, 0.039306640625, 0.0341796875, 0.0401611328125, 0.0183715820312, 0.0133972167969, -0.025390625, 0.00900268554688, -0.0059814453125, 0.0248107910156, 0.0177001953125, -0.00302124023438, 0.00396728515625, 0.00775146484375, 0.0476379394531, 0.0665588378906, 0.082763671875, 0.0417175292969, 0.0678405761719, 0.036376953125, 0.0613708496094}};

}; // end namespace
