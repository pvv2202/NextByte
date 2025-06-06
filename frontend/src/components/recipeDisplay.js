import { useState } from 'react';
import { TypeAnimation } from 'react-type-animation'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBookmark as faBookmarkSolid } from '@fortawesome/free-solid-svg-icons';
import { faBookmark as faBookmarkRegular } from '@fortawesome/free-regular-svg-icons';

function RecipeDisplay({recipe, setRecipe}) {
  const [saved, setSaved] = useState(false)
  let title = '';
  let ingredients = [];
  let directions = [];

  if (recipe) {
    title = recipe['title'];
    ingredients = recipe['ingredients'];
    directions = recipe['directions'];
  }
  
  return (
    <div className='w-full flex flex-col bg-white p-8 rounded-xl shadow-md'>
        <div className='flex gap-1 justify-between'>
          <h2 className='text-2xl'>
            <TypeAnimation 
              sequence={[title]} 
              speed={50}
              cursor={false}
            />
          </h2>
          <button className='hover:scale-105' onClick={() => setSaved(s => !s)}>
            {saved ? <FontAwesomeIcon icon={faBookmarkSolid} /> : <FontAwesomeIcon icon={faBookmarkRegular}/> }  
          </button>
          
        </div>

        <ul className='ml-4 mt-2 list-disc'>
          {ingredients.map((ing, idx)=>(
            <li key={idx}>
              <TypeAnimation 
                sequence={[500, ing]}
                speed={50}
                cursor={false}
              />
            </li>
          ))}
        </ul>
        
        <h2 className='text-xl mt-2 mb-2'>
          <TypeAnimation 
            sequence={[2000, "Directions"]}
            speed={50}
            cursor={false}
          />
        </h2>
        
        <ol className='ml-4 mt-2 list-decimal'>
          {directions.map((dir, idx)=>(
            <li key={idx}>
              <TypeAnimation 
                sequence={[3000, dir]}
                speed={50}
                cursor={false}
              />
            </li>
          ))}
        </ol>
        <div className='flex justify-end'>
          {recipe !== null ? <button className='w-1/4 mt-2 hover:scale-105 text-white rounded bg-gray-800' onClick={() => setRecipe(null)}>Clear</button> : ''}
        </div>
        
        
    </div>
  )
}

export default RecipeDisplay