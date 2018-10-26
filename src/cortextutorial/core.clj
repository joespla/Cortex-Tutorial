(ns cortextutorial.core
  (:require [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.optimize.adam :as adam]
            [cortextutorial.data :as data]
            [clojure.pprint :refer [pprint]])
  (:gen-class))

(defonce all-data (-> (data/load-data)
                      (shuffle)
                      (data/train-validation-split 0.70)))
(def num-nodes 32)

(def network-architecture
  [(layers/input 11 1 1 :id :x)

   (layers/linear num-nodes)
   (layers/relu)

   (layers/linear num-nodes)
   (layers/relu)

   (layers/linear 1 :id :y)])

(def starting-network {:network  (network/linear-network
                                   network-architecture)
                       :optimizer (adam/adam :alpha 0.01)})

(defn validate
  "Validación de la red"
  [cur-network]
  (println "Resultados:" (pr-str (data/stats (:network cur-network) (:validation all-data)))))

(defn train
  "Entrenamiento de la red"
  [incoming-network epoch-count]
  (loop [{:keys [network optimizer] :as cur-network} incoming-network
         epoch 0]


    (if (zero? (mod epoch 10))
      (println "Epoch: " epoch (pr-str (data/stats network (:train all-data)))))

    (if (> epoch-count epoch)
      (recur (execute/train network (:train all-data) :optimizer optimizer) (inc epoch))
      cur-network)))

(defn -main
  [& args]
  (println "Entenamiento en 100 epocas")
  (let [r1 (train starting-network 100)]
    (validate r1)))
