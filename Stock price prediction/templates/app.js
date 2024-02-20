async function predictPrice() {
    const stockCode = document.getElementById("stockCode").value;

    try {
        const response = await fetch(`/predict?stockCode=${stockCode}`);
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const result = await response.json();
        document.getElementById("result").innerHTML = `Predicted Price: ${result.predictedPrice}`;
    } catch (error) {
        console.error("Error:", error);
        document.getElementById("result").innerHTML = "Error predicting price.";
    }
}
