function sendQuery() {
    let query = document.getElementById("query").value;

    fetch("/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: query })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("response").innerText = data.response;
    });
}
