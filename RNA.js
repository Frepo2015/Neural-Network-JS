//Bibliotecas necesarias
const tf = require('@tensorflow/tfjs-node');
const f = require('fs');

let fileT = "SliceT-1.txt"; //Archivo que contienen los datos de entrenamiento
let fileP = "SliceP-1.txt";

//Importar datos de entrenamiento
arregloTrainning = leerTraining(fileT);   //Se llama a la Funcion para leer los datos de entrenamiento
numEjemplosT = parseInt(arregloTrainning[0]);       //Se guarda el numero de Ejemplos, Atributos y clases del conjunto de entrenamiento
numAtributosT = parseInt(arregloTrainning[1]);
numClasesT = parseInt(arregloTrainning[2]);
console.log("Numero de ejemplos (Train): " + numEjemplosT + "\n" +
    "Numero de atributos (Train): " + numAtributosT + "\n" +
    "Numero de clases (Train): " + numClasesT);//Se muestran las
let arregloXT = new Array(numEjemplosT);//Se declaran las matrices multidimenisionales
for (var i = 0; i < numEjemplosT; i++) {
    arregloXT[i] = arregloTrainning[i + 3].split(",").map(Number);
}
let arregloYT = new Array(numEjemplosT);
for (var i = 0; i < numEjemplosT; i++) {
    arregloYT[i] = arregloXT[i][numAtributosT];
    arregloXT[i].pop();
}
let XT = tf.tensor2d(arregloXT, [numEjemplosT, numAtributosT]);
let YT = tf.tensor2d(arregloYT, [numEjemplosT, 1]);
let YTc = tf.oneHot(tf.tensor1d(arregloYT, 'int32'), numClasesT);

console.log("Tamaño de la matriz de los datos de Entrenamiento: " + XT.shape);
console.log("Tamaño de la matriz de las clases esperadas de Entrenamiento: " + YT.shape);
console.log("Tamaño de la matriz de las clases esperadas clasificadas de Entrenamiento: " + YTc.shape + "\n\n");


//Importar datos de prueba
arregloPrueba = leerTraining(fileP);
//console.log(arregloPrueba[3]);
//let arregloTrainningsep=arregloTrainning[3].split(",");
numEjemplosP = parseInt(arregloPrueba[0]);
numAtributosP = parseInt(arregloPrueba[1]);
numClasesP = parseInt(arregloPrueba[2]);
console.log("Numero de ejemplos (Prueba): " + numEjemplosP + "\n" +
    "Numero de atributos (Prueba): " + numAtributosP + "\n" +
    "Numero de clases (Prueba): " + numClasesP);
let arregloXP = new Array(numEjemplosP);//Se declaran las matrices multidimenisionales
for (var i = 0; i < numEjemplosP; i++) {
    arregloXP[i] = arregloPrueba[i + 3].split(",").map(Number);
}
let arregloYP = new Array(numEjemplosP);
for (var i = 0; i < numEjemplosP; i++) {
    arregloYP[i] = arregloXP[i][numAtributosP];
    arregloXP[i].pop();
}
let XP = tf.tensor2d(arregloXP, [numEjemplosP, numAtributosP]);
let YP = tf.tensor2d(arregloYP, [numEjemplosP, 1]);
let YPc = tf.oneHot(tf.tensor1d(arregloYP, 'int32'), numClasesP);

console.log("Tamaño de la matriz de los datos de Prueba: " + XP.shape);
console.log("Tamaño de la matriz de las clases esperadas de Prueba: " + YP.shape);
console.log("Tamaño de la matriz de las clases esperadas clasificadas de Prueba: " + YPc.shape);


//TOPOLOGIA
var numCapas = 2;                               
var topologia = [numAtributosP,8, numClasesP] //Topologia
console.log(topologia);
createModel();
//Creamos la red
async function createModel() {

    model = tf.sequential();
    hidden1 = tf.layers.dense({
        units: 8,
        inputShape: [numAtributosP],
        activation: 'relu'
    });
    model.add(hidden1);
   /*hidden2 = tf.layers.dense({
        units: ,
        activation: 'sigmoid'
    });
    model.add(hidden2);
*/
    output = tf.layers.dense({
        units: numClasesP,
        activation: 'softmax'
    });
    model.add(output);
    //Definicion de parametros de entrenamiento y configuracion de la Red
    const learningRate = 0.05;
    const momentum = 0.099;
    const optimizador = tf.train.momentum(learningRate, momentum);
    model.compile({
        optimizer: optimizador,
        loss: 'meanSquaredError',
        metrics: ['accuracy'],
    });

    // Almacenar pesos
    await saveW("pesosIniciales.txt", numCapas, topologia, model);

    const settingTrain = {
        epochs: 50, //Épocas
        // validationSplit: 0.05,
        verbose: 1,

    };
    console.log("\n ENTRENANDO LA RED \n");
    const history = await model.fit(XT, YTc, settingTrain);
    //model.layers[0].getWeights()[0].print();
    //model.layers[1].getWeights()[0].print();
    //model.layers[2].getWeights()[0].print();
    //console.log(history);
    console.log("\n ENTRENAMIENTO FINALIZADO");
    console.log("\n\n---------------------EVALUANDO CONJUNTO DE PRUEBA------------------------------------\n\n")
    const prueba = await model.evaluate(XP, YPc, verbose = 1);//tensor
    const prueba1 = prueba[0];
    const prueba2 = prueba[1];
    const valorLoss = prueba1.dataSync();
    const valorAcc = Math.round(((prueba2.dataSync()) * 100));
    console.log("\nLoss del conjunto de Prueba: " + valorLoss);
    console.log("\nAcc del conjunto de Prueba: " + valorAcc + "%\n");
}
async function saveW(file, nCapas, topologia, modelo ){
    console.log("Guardando Pesos...\n")
    f.open(file, 'w', function (err, fd) {
        if (err) {
            throw 'could not open file: ' + err;
        }
        let buffer = new Buffer.from(nCapas.toString() + "\n");
        f.write(fd, buffer, 0, buffer.length, null, function (err) {
            if (err) throw 'error writing file: ' + err;
            f.close(fd, function () {
            });
        });
        for(var j=0;j<=nCapas;j++){
            var t=topologia[j];
            let buffer1 = new Buffer.from(t + "\n");
            f.write(fd, buffer1, 0, buffer1.length, null, function (err) {
                if (err) throw 'error writing file: ' + err;
                f.close(fd, function () {
                });
            });
        }
        for (var i = 0; i < nCapas; i++) {
            const tensorCapa1 = modelo.layers[i].getWeights()[0];//Tensor
            console.log("\nTamaño Tensor en la capa"+(i+1)+" : "+tensorCapa1.shape);
            const valoresCapa1 = tensorCapa1.dataSync();
            const matrizPesosC1 = Array.from(valoresCapa1);
            //console.log(matrizPesosC1);
            let buffer3 = new Buffer.from(matrizPesosC1 + "\n");
            f.write(fd, buffer3, 0, buffer3.length, null, function (err) {
                if (err) throw 'error writing file: ' + err;
                f.close(fd, function () {
                });
            });
            const sesgoCapa1 = modelo.layers[i].getWeights()[1];//Tensor
            console.log("\nTamaño Tensor (Sesgo) en la capa"+(i+1)+" : "+sesgoCapa1.shape);
            const valoresSesgo1 = sesgoCapa1.dataSync();
            const matrizSesgosC1 = Array.from(valoresSesgo1);
            //console.log(matrizPesosC1);
            let buffer4 = new Buffer.from(matrizSesgosC1 + "\n");
            f.write(fd, buffer4, 0, buffer4.length, null, function (err) {
                if (err) throw 'error writing file: ' + err;
                f.close(fd, function () {
                });
            });
        }
        console.log("Pesos iniciales guardados.")
    });
}
//Funcion para leer el Archivo y dividirlo por renglones.
function leerTraining(file) {
    const contents = f.readFileSync(file, 'utf-8');
    const arr = contents.split(/\r?\n/);
    return arr;
}

//METAHEURISTICA***********************************************************************
