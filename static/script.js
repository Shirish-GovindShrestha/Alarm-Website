function addMessage(sender, text, sql="") {
    const chat = document.getElementById("chat");
    const p = document.createElement("p");
    p.className = sender;
    p.innerText = text;
    chat.appendChild(p);
    
    if (sql) {
        const sqlElem = document.createElement("div");
        sqlElem.className = "sql";
        sqlElem.innerText = sql;
        chat.appendChild(sqlElem);
    }

    chat.scrollTop = chat.scrollHeight;
}

function updateAlarms(alarms) {
    const list = document.getElementById("alarms");
    list.innerHTML = "";
    if (alarms.length === 0) return;
    alarms.forEach(a => {
        const li = document.createElement("li");
        li.innerText = `${a.label} at ${a.time}`;
        list.appendChild(li);
    });
}

async function sendQuery() {
    const input = document.getElementById("query");
    const query = input.value.trim();
    if (!query) return;
    
    addMessage("user", query);
    input.value = "";

    try {
        const res = await fetch("/alarm", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query })
        });
        const data = await res.json();
        addMessage("bot", data.response, data.sql_query);
        updateAlarms(data.alarms);
    } catch (err) {
        addMessage("bot", "Error contacting server.");
        console.error(err);
    }
}

// Allow pressing Enter to send
document.getElementById("query").addEventListener("keypress", function(e) {
    if (e.key === "Enter") sendQuery();
});
