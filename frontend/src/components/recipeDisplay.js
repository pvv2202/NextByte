import React from 'react'
import { TypeAnimation } from 'react-type-animation'

function RecipeDisplay({recipe, setRecipe}) {
  
  let title = '';
  let ingredients = [];
  let directions = [];

  if (recipe) {
    title = recipe['title'];
    ingredients = recipe['ingredients'];
    directions = recipe['directions'];
  }
  
  return (
    <div className='w-full flex flex-col bg-white p-8 rounded shadow-md'>
        <div className='flex justify-between'>
          <h2 className='text-2xl'>
            <TypeAnimation 
              sequence={[title, 1000]} 
              speed={50}
              cursor={false}
            />
          </h2>
          <img src="nextbyte.png" className='w-10 object-contain' alt="nextbyte" />
        </div>
        
        <h2 className='text-xl mt-2 mb-2'>Ingredients:</h2>
        <ul className='ml-4 list-disc'>
          {ingredients.map((ing, idx)=>(
            <li key={idx}>
              <TypeAnimation 
                sequence={[ing, 1000]}
                speed={50}
                cursor={false}
              />
            </li>
          ))}
        </ul>
        <h2 className='text-xl mt-2 mb-2'>Directions</h2>
        <ol className='ml-4 list-decimal'>
          {directions.map((dir, idx)=>(
            <li key={idx}>
              <TypeAnimation 
                sequence={[dir, 1000]}
                speed={50}
                cursor={false}
              />
            </li>
          ))}
        </ol>
        <div className='flex justify-end'>
          {recipe !== null ? <button className='w-1/4 mt-2 hover:scale-105 text-white rounded bg-gray-800' onClick={() => setRecipe(null)}>Delete</button> : ''}
        </div>
        
        
    </div>
  )
}

export default RecipeDisplay