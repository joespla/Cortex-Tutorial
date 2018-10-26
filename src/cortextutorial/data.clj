(ns cortextutorial.data
  (:require [clojure.string :as string]
            [cortex.nn.execute :as execute]))

;;Convierte String a Float
(defn parse-float [s]
  (Float/parseFloat s))

;;Carga el archivo
(defn load-data
  "Carga Archivo"
  []
  (as-> (slurp "resources/data.csv") d
        (string/split d #"\n")
        (map #(string/split % #",") d)
        (map #(map parse-float %) d)
        (map (fn [s] {:x (drop-last s) :y (last s)}) d)))

;;calcula el error
(defn calc-error
  [f as bs]
  (as-> (map #(f (- %1 %2)) as bs) d
        (apply + d)
        (/ d (count as))))

;;Calcula el error cuadratico
(defn mean-squared-error
  [as bs]
  (calc-error #(* % %) as bs))

(defn mean-error
  "Promedio del error"
  [as bs]
  (calc-error #(Math/abs %) as bs))

(defn train-validation-split
  "Parte el dataset"
  [d percent]
  (let [c (int (* percent (count d)))]
    {:train (take c d)
     :validation (drop c d)}))

(defn stats
  "Muestra estadisticas basicas."
  [network data]
  (let [results (execute/run network (map #(dissoc % :y) data))
        predictions (mapcat :y results)
        actual-vals (map :y data)]
    {:mean-squared-error (mean-squared-error predictions actual-vals)
     :mean-error (mean-error predictions actual-vals)}))