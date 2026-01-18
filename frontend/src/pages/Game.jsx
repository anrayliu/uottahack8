import React, { useState } from "react";

const Game = () => {
  const [selectedOption, setSelectedOption] = useState("");

  return (
    <div className="min-h-screen bg-neutral-950 text-white p-8 flex flex-col items-center justify-center">
      <h1 className="text-4xl font-black tracking-tight mb-8">Game</h1>

      <div className="w-64">
        <label className="block text-neutral-400 text-sm font-medium mb-2">
          Select an option
        </label>
        <select
          value={selectedOption}
          onChange={(e) => setSelectedOption(e.target.value)}
          className="w-full bg-neutral-800 border border-neutral-700 text-white rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-amber-500 focus:border-transparent"
        >
          <option value="">Choose...</option>
          <option value="1">1</option>
          <option value="2">2</option>
          <option value="3">3</option>
          <option value="4">4</option>
        </select>
      </div>
    </div>
  );
};

export default Game;
