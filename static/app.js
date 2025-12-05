document.addEventListener('DOMContentLoaded', () => {
    const runBtn = document.getElementById('run-btn');
    const prePromptInput = document.getElementById('pre-prompt');
    const progressContainer = document.getElementById('progress-container');
    const errorBanner = document.getElementById('error-banner');
    const resultsSection = document.getElementById('results-section');

    // Results elements
    const baseAcc = document.getElementById('base-acc');
    const baseCorrect = document.getElementById('base-correct');
    const baseCost = document.getElementById('base-cost');
    const baseCpc = document.getElementById('base-cpc');

    const scaffAcc = document.getElementById('scaff-acc');
    const scaffCorrect = document.getElementById('scaff-correct');
    const scaffCost = document.getElementById('scaff-cost');
    const scaffCpc = document.getElementById('scaff-cpc');

    const deltaAcc = document.getElementById('delta-acc');
    const deltaEff = document.getElementById('delta-eff');

    runBtn.addEventListener('click', async () => {
        // Reset UI
        errorBanner.classList.add('hidden');
        resultsSection.classList.add('hidden');
        runBtn.disabled = true;
        prePromptInput.disabled = true;
        document.querySelectorAll('input[type="radio"]').forEach(r => r.disabled = true);

        progressContainer.classList.remove('hidden');

        // Get values
        const prePrompt = prePromptInput.value;
        const model = document.querySelector('input[name="model"]:checked').value;
        const benchmark = document.querySelector('input[name="benchmark"]:checked').value;

        try {
            const response = await fetch('/api/run-test', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    pre_prompt: prePrompt,
                    model: model,
                    benchmark: benchmark
                })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to run test');
            }

            // Handle streaming response
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.trim()) continue;

                    try {
                        const event = JSON.parse(line);

                        if (event.type === 'progress') {
                            const pct = Math.round((event.completed / event.total) * 100);
                            progressContainer.querySelector('#progress-text').textContent = event.message;
                            progressContainer.querySelector('#progress-fill').style.width = `${pct}%`;
                        } else if (event.type === 'result') {
                            displayResults(event.data);
                        }
                    } catch (e) {
                        console.error('Error parsing JSON line:', e);
                    }
                }
            }

        } catch (error) {
            showError(error.message);
        } finally {
            // Re-enable UI
            runBtn.disabled = false;
            prePromptInput.disabled = false;
            document.querySelectorAll('input[name="model"]').forEach(r => r.disabled = false);
            document.querySelector('input[name="benchmark"][value="mmlu-pro"]').disabled = false;
            progressContainer.classList.add('hidden');
        }
    });

    function displayResults(data) {
        // Baseline
        baseAcc.textContent = `${data.baseline.accuracy_pct}%`;
        baseCorrect.textContent = `${data.baseline.correct}/${data.baseline.total_questions} correct`;
        baseCost.textContent = `$${data.baseline.total_cost_usd.toFixed(4)}`;
        baseCpc.textContent = `$${data.baseline.cost_per_correct_usd.toFixed(4)}`;

        // Scaffolded
        scaffAcc.textContent = `${data.scaffolded.accuracy_pct}%`;
        scaffCorrect.textContent = `${data.scaffolded.correct}/${data.scaffolded.total_questions} correct`;
        scaffCost.textContent = `$${data.scaffolded.total_cost_usd.toFixed(4)}`;
        scaffCpc.textContent = `$${data.scaffolded.cost_per_correct_usd.toFixed(4)}`;

        // Deltas
        const accDelta = (data.scaffolded.accuracy_pct - data.baseline.accuracy_pct).toFixed(1);
        const effDelta = (data.scaffolded.cost_per_correct_usd - data.baseline.cost_per_correct_usd).toFixed(4);

        deltaAcc.textContent = `${accDelta >= 0 ? '+' : ''}${accDelta}%`;
        deltaAcc.className = `delta-value ${accDelta >= 0 ? 'positive' : 'negative'}`;

        deltaEff.textContent = `${effDelta >= 0 ? '+' : ''}$${effDelta}`;
        // For cost efficiency, negative delta (lower cost) is usually better, but here we track cost per correct.
        // Lower cost per correct is better. So negative delta is green.
        deltaEff.className = `delta-value ${effDelta <= 0 ? 'positive' : 'negative'}`;

        resultsSection.classList.remove('hidden');
    }

    function showError(message) {
        errorBanner.textContent = `Error: ${message}`;
        errorBanner.classList.remove('hidden');
    }
});
